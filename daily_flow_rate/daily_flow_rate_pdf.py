"""Recognize scanned daily mean flow tables arranged by day and month.

The supported form has one day column followed by twelve month columns.  The
daily body has no horizontal rules, so row order is reconstructed from real
ink anchors instead of fitting a fixed pitch.  Each selected row anchor must
be supported by several columns and is checked against the printed 1-31 day
sequence before the 12 month cells are recognized.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

# Cached models must remain usable when the machine has no model-source probe.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import TextRecognition
from scipy.interpolate import PchipInterpolator

from flood_hydro_factor_pdf import log, safe_filename, verify_device
from reconstruct_grid_table import prepare_cell, result_value


MONTH_NAMES = [
    "一月",
    "二月",
    "三月",
    "四月",
    "五月",
    "六月",
    "七月",
    "八月",
    "九月",
    "十月",
    "十一月",
    "十二月",
]
COMMON_MONTH_LENGTHS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
LEAP_MONTH_LENGTHS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
EXPECTED_COLUMN_COUNT = 13
EXPECTED_VERTICAL_RULE_COUNT = EXPECTED_COLUMN_COUNT + 1
EXPECTED_DAY_COUNT = 31

NUMERIC_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "C": "0",
        "c": "0",
        "D": "0",
        "U": "0",
        "u": "0",
        "V": "0",
        "v": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "|": "1",
        "!": "1",
        "J": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "G": "6",
        "T": "7",
        "t": "7",
        "B": "8",
        "b": "8",
        "g": "9",
        "q": "9",
    }
)


@dataclass
class AnchorCandidate:
    top: int
    bottom: int
    center: float
    ink_centroid: float
    support_columns: int
    day_column_support: bool
    score: float


@dataclass
class PageGeometry:
    page: int
    table_index: int
    width: int
    height: int
    table_left: int
    table_top: int
    table_right: int
    header_bottom: int
    statistics_top: int
    vertical_rules: list[int]
    horizontal_rules: list[int]
    candidate_anchors: list[AnchorCandidate]
    row_anchors: list[AnchorCandidate]
    anchor_method: str = "day-column-projection"
    column_center_x: list[float] = field(default_factory=list)
    row_curve_centers: list[list[float]] = field(default_factory=list)
    table_top_curve: list[float] = field(default_factory=list)
    header_curve: list[float] = field(default_factory=list)
    statistics_curve: list[float] = field(default_factory=list)
    vertical_rule_models: list[list[float]] = field(default_factory=list)
    month_decimal_anchors: list[list[dict[str, Any]]] = field(
        default_factory=list
    )
    virtual_month_anchors: list[dict[str, Any]] = field(default_factory=list)
    month_lengths: list[int] = field(default_factory=list)


@dataclass
class TableRegion:
    """A table located on the photographed source page."""

    page: int
    table_index: int
    source_bbox: tuple[int, int, int, int]
    source_rules: list[int]
    source_components: list[dict[str, float]]
    image: np.ndarray


def clustered_positions(values: list[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if not groups or value - float(np.mean(groups[-1])) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.mean(group)) for group in groups]


def render_pdf_page(pdf_path: Path, page_index: int, width: int) -> np.ndarray:
    document = pdfium.PdfDocument(str(pdf_path))
    page = document[page_index]
    scale = width / page.get_width()
    image = page.render(scale=scale).to_pil().convert("RGB")
    page.close()
    document.close()
    array = np.asarray(image)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def adaptive_ink(gray: np.ndarray) -> np.ndarray:
    """Retain faint printing under uneven illumination in photographed pages."""

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        13,
    )


def detect_table_regions(image: np.ndarray, page_number: int) -> list[TableRegion]:
    """Find every 14-rule daily-flow table on one page."""

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = adaptive_ink(gray)
    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(35, round(height * 0.021)))
        ),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
    components: list[dict[str, float]] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if (
            height * 0.18 <= component_height <= height * 0.65
            and component_width <= max(22, width * 0.020)
            and area >= component_height * 0.35
        ):
            components.append(
                {
                    "x": float(x + component_width / 2),
                    "y": float(y),
                    "height": float(component_height),
                    "bottom": float(y + component_height - 1),
                }
            )

    groups: list[list[dict[str, float]]] = []
    for component in sorted(components, key=lambda item: item["y"]):
        if (
            not groups
            or abs(
                component["y"]
                - float(np.median([item["y"] for item in groups[-1]]))
            )
            > height * 0.035
        ):
            groups.append([component])
        else:
            groups[-1].append(component)

    regions: list[TableRegion] = []
    for group in groups:
        try:
            rules = select_regular_vertical_rules(group, width)
        except RuntimeError:
            continue
        selected = [
            min(group, key=lambda item, x=x: abs(item["x"] - x))
            for x in rules
        ]
        if len({round(item["x"]) for item in selected}) != EXPECTED_VERTICAL_RULE_COUNT:
            continue
        top = max(0, round(min(item["y"] for item in selected)))
        outer_bottom = max(selected[0]["bottom"], selected[-1]["bottom"])
        bottom = min(height - 1, round(outer_bottom))
        regions.append(
            TableRegion(
                page=page_number,
                table_index=0,
                source_bbox=(rules[0], top, rules[-1], bottom),
                source_rules=rules,
                source_components=selected,
                image=np.empty((0, 0, 3), dtype=np.uint8),
            )
        )
    regions.sort(key=lambda item: item.source_bbox[1])
    for table_index, region in enumerate(regions, start=1):
        region.table_index = table_index
    return regions


def rectify_table_region(image: np.ndarray, region: TableRegion) -> np.ndarray:
    """Flatten the bowed upper border and column rules of a photographed table."""

    left, _, right, _ = region.source_bbox
    components = region.source_components
    xs = np.asarray([item["x"] for item in components], dtype=np.float32)
    tops = np.asarray([item["y"] for item in components], dtype=np.float32)
    top_coefficients = np.polyfit(xs, tops, 2)
    output_width = right - left + 1
    output_x = np.arange(output_width, dtype=np.float32)
    source_x = output_x + left
    top_curve = np.polyval(top_coefficients, source_x).astype(np.float32)
    bottom_curve = np.interp(
        source_x,
        [components[0]["x"], components[-1]["x"]],
        [components[0]["bottom"], components[-1]["bottom"]],
    ).astype(np.float32)
    output_height = max(300, round(float(np.median(bottom_curve - top_curve))))
    fractions = np.linspace(0.0, 1.0, output_height, dtype=np.float32)[:, None]
    map_x = np.broadcast_to(source_x[None, :], (output_height, output_width)).copy()
    map_y = top_curve[None, :] + fractions * (bottom_curve - top_curve)[None, :]
    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 31)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=245)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def select_regular_vertical_rules(
    components: list[dict[str, float]], image_width: int
) -> list[int]:
    tolerance = max(4.0, image_width * 0.0035)
    x_positions = clustered_positions(
        [item["x"] for item in components], tolerance=tolerance
    )
    if len(x_positions) < EXPECTED_VERTICAL_RULE_COUNT:
        raise RuntimeError(
            f"仅检测到{len(x_positions)}条长竖线，"
            f"日流量表需要{EXPECTED_VERTICAL_RULE_COUNT}条。"
        )

    candidates: list[tuple[float, list[float]]] = []
    for start in range(
        len(x_positions) - EXPECTED_VERTICAL_RULE_COUNT + 1
    ):
        rules = x_positions[
            start : start + EXPECTED_VERTICAL_RULE_COUNT
        ]
        gaps = np.diff(np.asarray(rules, dtype=float))
        mean_gap = float(np.mean(gaps))
        if mean_gap <= image_width * 0.025:
            continue
        span = rules[-1] - rules[0]
        if span < image_width * 0.55:
            continue
        cv = float(np.std(gaps) / max(mean_gap, 1.0))
        edge_penalty = abs(span / image_width - 0.79) * 0.12
        candidates.append((cv + edge_penalty, rules))
    if not candidates:
        raise RuntimeError("没有找到由14条近似等距竖线组成的日流量表。")
    candidates.sort(key=lambda item: item[0])
    chosen = candidates[0][1]
    gaps = np.diff(np.asarray(chosen, dtype=float))
    if float(np.max(gaps) / max(np.min(gaps), 1.0)) > 1.35:
        raise RuntimeError("日流量表竖线间距不稳定，不能安全切分13列。")
    return [round(value) for value in chosen]


def detect_rule_geometry(image: np.ndarray) -> tuple[list[int], list[int]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]

    vertical = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(70, round(height * 0.043)))
        ),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
    components: list[dict[str, float]] = []
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if (
            component_height >= height * 0.23
            and component_width <= max(14, width * 0.012)
            and area >= component_height * 0.55
        ):
            components.append(
                {
                    "x": float(x + component_width / 2),
                    "y": float(y),
                    "height": float(component_height),
                }
            )
    vertical_rules = select_regular_vertical_rules(components, width)

    table_span = vertical_rules[-1] - vertical_rules[0]
    horizontal = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(140, round(table_span * 0.18)), 1)
        ),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, 8)
    horizontal_y: list[float] = []
    for index in range(1, count):
        x, y, component_width, component_height, _ = stats[index]
        if (
            component_width >= table_span * 0.78
            and component_height <= max(12, height * 0.008)
            and x <= vertical_rules[0] + table_span * 0.08
            and x + component_width
            >= vertical_rules[-1] - table_span * 0.08
        ):
            horizontal_y.append(float(y + component_height / 2))
    horizontal_rules = [
        round(value)
        for value in clustered_positions(
            horizontal_y, tolerance=max(3.0, height * 0.002)
        )
    ]
    if len(horizontal_rules) < 3:
        raise RuntimeError("没有检测到表头、正文和统计区所需的长横线。")
    return vertical_rules, horizontal_rules


def find_table_vertical_bounds(
    image: np.ndarray, vertical_rules: list[int]
) -> tuple[int, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    left = max(0, vertical_rules[0] - 3)
    right = min(image.shape[1], vertical_rules[-1] + 4)
    region = ink[:, left:right]
    vertical = cv2.morphologyEx(
        region,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(70, round(image.shape[0] * 0.043)))
        ),
    )
    ys, _ = np.nonzero(vertical)
    if ys.size == 0:
        raise RuntimeError("无法确定日流量表竖线的上下范围。")
    return int(np.min(ys)), int(np.max(ys))


def active_bands(active: np.ndarray, offset: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active.tolist()):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            if index - start >= 2:
                bands.append((offset + start, offset + index - 1))
            start = None
    if start is not None and len(active) - start >= 2:
        bands.append((offset + start, offset + len(active) - 1))
    return bands


def column_support(
    body_ink: np.ndarray,
    vertical_rules: list[int],
    body_left: int,
    body_top: int,
    top: int,
    bottom: int,
) -> tuple[int, bool]:
    local_top = max(0, top - body_top)
    local_bottom = min(body_ink.shape[0], bottom - body_top + 1)
    support = 0
    day_support = False
    for column in range(EXPECTED_COLUMN_COUNT):
        left = max(0, vertical_rules[column] - body_left + 3)
        right = min(
            body_ink.shape[1],
            vertical_rules[column + 1] - body_left - 3,
        )
        cell = body_ink[local_top:local_bottom, left:right]
        area = int(np.count_nonzero(cell))
        required = max(4, round(cell.size * 0.006))
        if area >= required:
            support += 1
            if column == 0:
                day_support = True
    return support, day_support


def build_anchor_candidates(
    image: np.ndarray,
    vertical_rules: list[int],
    body_top: int,
    body_bottom: int,
) -> list[AnchorCandidate]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fixed_ink = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    otsu_ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    body_left = vertical_rules[0]
    body_right = vertical_rules[-1]
    choices: list[tuple[float, np.ndarray, np.ndarray, list[tuple[int, int]]]] = []
    threshold_fractions = (
        0.009,
        0.011,
        0.013,
        0.015,
        0.020,
        0.025,
        0.030,
        0.035,
        0.040,
        0.045,
        0.050,
        0.060,
        0.070,
        0.080,
    )
    for page_ink in (fixed_ink, otsu_ink):
        body = page_ink[body_top:body_bottom, body_left:body_right].copy()
        vertical_mask = cv2.morphologyEx(
            body,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (1, max(40, round((body_bottom - body_top) * 0.13))),
            ),
        )
        body[vertical_mask > 0] = 0
        projection = np.count_nonzero(body, axis=1)
        for fraction in threshold_fractions:
            threshold = max(10, round(body.shape[1] * fraction))
            bands = active_bands(projection >= threshold, body_top)
            bands = [
                band
                for band in bands
                if 2 <= band[1] - band[0] + 1 <= image.shape[0] * 0.030
            ]
            choices.append((fraction, body, projection, bands))
    _, body, projection, chosen_bands = min(
        choices,
        key=lambda item: (
            abs(len(item[3]) - EXPECTED_DAY_COUNT),
            0 if len(item[3]) >= EXPECTED_DAY_COUNT else 1,
            abs(item[0] - 0.025),
        ),
    )

    candidates: list[AnchorCandidate] = []
    for top, bottom in chosen_bands:
        local_top = top - body_top
        local_bottom = bottom - body_top + 1
        weights = projection[local_top:local_bottom].astype(float)
        y_values = np.arange(top, bottom + 1, dtype=float)
        ink_centroid = float(
            np.average(y_values, weights=weights)
            if float(np.sum(weights)) > 0
            else (top + bottom) / 2
        )
        support, day_support = column_support(
            body,
            vertical_rules,
            body_left,
            body_top,
            top,
            bottom,
        )
        score = support + (3.0 if day_support else 0.0)
        candidates.append(
            AnchorCandidate(
                top=top,
                bottom=bottom,
                center=(top + bottom) / 2,
                ink_centroid=ink_centroid,
                support_columns=support,
                day_column_support=day_support,
                score=score,
            )
        )
    return candidates


def build_day_column_candidates(
    image: np.ndarray,
    vertical_rules: list[int],
    body_top: int,
    body_bottom: int,
) -> list[AnchorCandidate]:
    """Use the printed day numbers as a strong row-order anchor after dewarping."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    body_left = vertical_rules[0]
    body_right = vertical_rules[-1]
    body = page_ink[body_top:body_bottom, body_left:body_right].copy()
    vertical_mask = cv2.morphologyEx(
        body,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(40, round(body.shape[0] * 0.13)))
        ),
    )
    body[vertical_mask > 0] = 0
    day_left = max(0, vertical_rules[0] - body_left + 3)
    day_right = min(
        body.shape[1], vertical_rules[1] - body_left - 3
    )
    projection = np.count_nonzero(body[:, day_left:day_right], axis=1)
    choices: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for threshold in range(1, 9):
        bands = active_bands(projection >= threshold, body_top)
        bands = [
            band
            for band in bands
            if 2 <= band[1] - band[0] + 1 <= image.shape[0] * 0.030
        ]
        choices.append(
            (
                abs(len(bands) - EXPECTED_DAY_COUNT),
                0 if len(bands) >= EXPECTED_DAY_COUNT else 1,
                threshold,
                bands,
            )
        )
    _, _, _, chosen_bands = min(choices)
    candidates: list[AnchorCandidate] = []
    for top, bottom in chosen_bands:
        local_top = top - body_top
        local_bottom = bottom - body_top + 1
        weights = projection[local_top:local_bottom].astype(float)
        y_values = np.arange(top, bottom + 1, dtype=float)
        ink_centroid = float(
            np.average(y_values, weights=weights)
            if float(np.sum(weights)) > 0
            else (top + bottom) / 2
        )
        support, day_support = column_support(
            body,
            vertical_rules,
            body_left,
            body_top,
            top,
            bottom,
        )
        candidates.append(
            AnchorCandidate(
                top=top,
                bottom=bottom,
                center=(top + bottom) / 2,
                ink_centroid=ink_centroid,
                support_columns=support,
                day_column_support=day_support,
                score=support + (3.0 if day_support else 0.0),
            )
        )
    return candidates


def build_multicolumn_support_candidates(
    image: np.ndarray,
    vertical_rules: list[int],
    body_top: int,
    body_bottom: int,
) -> list[AnchorCandidate]:
    """Recover row anchors from ink that agrees across many month columns.

    Photograph noise can join or split digits in the narrow day column.  A true
    data row, however, normally contains ink at nearly the same y position in
    most of the 13 columns.  This fallback counts that cross-column agreement
    and therefore does not assume that a row boundary is perfectly horizontal.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    body_left = vertical_rules[0]
    body_right = vertical_rules[-1]
    body = page_ink[body_top:body_bottom, body_left:body_right].copy()
    vertical_mask = cv2.morphologyEx(
        body,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, max(40, round(body.shape[0] * 0.13)))
        ),
    )
    body[vertical_mask > 0] = 0

    projections: list[np.ndarray] = []
    for column in range(EXPECTED_COLUMN_COUNT):
        left = max(0, vertical_rules[column] - body_left + 5)
        right = min(
            body.shape[1], vertical_rules[column + 1] - body_left - 5
        )
        projections.append(np.count_nonzero(body[:, left:right], axis=1))
    column_projections = np.asarray(projections, dtype=np.int32)

    option_candidates: list[
        tuple[tuple[float, int, int, int], list[AnchorCandidate]]
    ] = []
    closest_candidates: tuple[
        tuple[int, int, int, int], list[AnchorCandidate]
    ] | None = None
    for pixel_threshold in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        support_profile = np.count_nonzero(
            column_projections >= pixel_threshold, axis=0
        )
        for required_columns in (8, 7, 9, 6, 10):
            bands = active_bands(
                support_profile >= required_columns, body_top
            )
            bands = [
                band
                for band in bands
                if 2 <= band[1] - band[0] + 1 <= image.shape[0] * 0.030
            ]
            candidates: list[AnchorCandidate] = []
            for top, bottom in bands:
                local_top = top - body_top
                local_bottom = bottom - body_top + 1
                weights = support_profile[local_top:local_bottom].astype(float)
                y_values = np.arange(top, bottom + 1, dtype=float)
                ink_centroid = float(
                    np.average(y_values, weights=weights)
                    if float(np.sum(weights)) > 0
                    else (top + bottom) / 2
                )
                support, day_support = column_support(
                    body,
                    vertical_rules,
                    body_left,
                    body_top,
                    top,
                    bottom,
                )
                candidates.append(
                    AnchorCandidate(
                        top=top,
                        bottom=bottom,
                        center=(top + bottom) / 2,
                        ink_centroid=ink_centroid,
                        support_columns=support,
                        day_column_support=day_support,
                        score=support + (3.0 if day_support else 0.0),
                    )
                )

            closest_key = (
                abs(len(candidates) - EXPECTED_DAY_COUNT),
                0 if len(candidates) >= EXPECTED_DAY_COUNT else 1,
                abs(required_columns - 8),
                pixel_threshold,
            )
            if closest_candidates is None or closest_key < closest_candidates[0]:
                closest_candidates = (closest_key, candidates)

            try:
                selected = choose_31_anchors(
                    candidates, body_top, body_bottom
                )
            except RuntimeError:
                continue

            centers = np.asarray(
                [candidate.center for candidate in selected], dtype=float
            )
            # The printed layout inserts a wider gap after days 5, 10, ... 25.
            # Fitting that known rhythm scores evidence; it never fabricates a row.
            pattern = np.asarray(
                [index + 0.82 * (index // 5) for index in range(31)],
                dtype=float,
            )
            slope, intercept = np.polyfit(pattern, centers, 1)
            residual = float(
                np.sqrt(np.mean((centers - (slope * pattern + intercept)) ** 2))
                / max(abs(slope), 1.0)
            )
            option_candidates.append(
                (
                    (
                        residual,
                        abs(len(candidates) - EXPECTED_DAY_COUNT),
                        abs(required_columns - 8),
                        pixel_threshold,
                    ),
                    candidates,
                )
            )

    if option_candidates:
        return min(option_candidates, key=lambda item: item[0])[1]
    if closest_candidates is not None:
        return closest_candidates[1]
    return []


def choose_31_anchors(
    candidates: list[AnchorCandidate], body_top: int, body_bottom: int
) -> list[AnchorCandidate]:
    if len(candidates) < EXPECTED_DAY_COUNT:
        raise RuntimeError(
            f"墨迹投影只形成{len(candidates)}个候选行，少于31行；"
            "拒绝按固定行距凭空补行。"
        )
    if len(candidates) == EXPECTED_DAY_COUNT:
        chosen = candidates
    else:
        centers = np.asarray([item.center for item in candidates], dtype=float)
        gaps = np.diff(centers)
        lower_gaps = gaps[gaps <= np.percentile(gaps, 70)]
        base_pitch = float(np.median(lower_gaps)) if lower_gaps.size else 1.0
        base_pitch = max(base_pitch, (body_bottom - body_top) / 55)
        negative = -1.0e12
        n = len(candidates)
        dp = np.full((EXPECTED_DAY_COUNT, n), negative, dtype=float)
        parent = np.full((EXPECTED_DAY_COUNT, n), -1, dtype=int)
        for index, candidate in enumerate(candidates):
            start_penalty = abs(candidate.center - body_top) / base_pitch * 0.12
            dp[0, index] = candidate.score - start_penalty
        group_breaks = {5, 10, 15, 20, 25}
        for selected_index in range(1, EXPECTED_DAY_COUNT):
            previous_day = selected_index
            expected_gap = base_pitch * (
                1.82 if previous_day in group_breaks else 1.0
            )
            for current in range(selected_index, n):
                for previous in range(selected_index - 1, current):
                    gap = candidates[current].center - candidates[previous].center
                    gap_penalty = abs(gap - expected_gap) / base_pitch * 2.2
                    value = (
                        dp[selected_index - 1, previous]
                        + candidates[current].score
                        - gap_penalty
                    )
                    if value > dp[selected_index, current]:
                        dp[selected_index, current] = value
                        parent[selected_index, current] = previous
        end_scores = dp[-1].copy()
        for index, candidate in enumerate(candidates):
            end_scores[index] -= (
                abs(body_bottom - candidate.center) / base_pitch * 0.12
            )
        end = int(np.argmax(end_scores))
        if end_scores[end] <= negative / 2:
            raise RuntimeError("31行锚点动态规划没有形成有效路径。")
        indices = [end]
        for selected_index in range(EXPECTED_DAY_COUNT - 1, 0, -1):
            end = int(parent[selected_index, end])
            if end < 0:
                raise RuntimeError("31行锚点动态规划回溯失败。")
            indices.append(end)
        chosen = [candidates[index] for index in reversed(indices)]

    if len(chosen) != EXPECTED_DAY_COUNT:
        raise RuntimeError("最终行锚点数量不是31。")
    if any(
        chosen[index].center >= chosen[index + 1].center
        for index in range(len(chosen) - 1)
    ):
        raise RuntimeError("最终行锚点没有严格递增。")
    weak = [
        index + 1
        for index, item in enumerate(chosen)
        if item.support_columns < 5 or not item.day_column_support
    ]
    if weak:
        raise RuntimeError(
            "以下日序锚点缺少足够的多列/日列证据："
            + "、".join(map(str, weak))
        )
    return chosen


def detect_page_geometry(
    image: np.ndarray, page_number: int, table_index: int = 1
) -> PageGeometry:
    vertical_rules, horizontal_rules = detect_rule_geometry(image)
    vertical_top, _ = find_table_vertical_bounds(image, vertical_rules)
    table_top = min(
        horizontal_rules,
        key=lambda value: abs(value - vertical_top),
    )
    rules_below_top = [
        value
        for value in horizontal_rules
        if value > table_top + image.shape[0] * 0.012
    ]
    if len(rules_below_top) < 2:
        raise RuntimeError("无法确定表头底线和统计区起始线。")
    header_bottom = rules_below_top[0]
    statistics_candidates = [
        value
        for value in rules_below_top[1:]
        if value > header_bottom + image.shape[0] * 0.20
    ]
    if not statistics_candidates:
        raise RuntimeError("无法确定逐日数据与统计区之间的分界线。")
    statistics_top = statistics_candidates[0]
    candidates = build_anchor_candidates(
        image,
        vertical_rules,
        header_bottom + 2,
        statistics_top - 1,
    )
    anchors = choose_31_anchors(
        candidates, header_bottom + 2, statistics_top - 1
    )
    return PageGeometry(
        page=page_number,
        table_index=table_index,
        width=image.shape[1],
        height=image.shape[0],
        table_left=vertical_rules[0],
        table_top=table_top,
        table_right=vertical_rules[-1],
        header_bottom=header_bottom,
        statistics_top=statistics_top,
        vertical_rules=vertical_rules,
        horizontal_rules=horizontal_rules,
        candidate_anchors=candidates,
        row_anchors=anchors,
    )


def darkest_rule_row(gray: np.ndarray, top: int, bottom: int) -> int:
    top = max(0, top)
    bottom = min(gray.shape[0], bottom)
    if bottom <= top:
        raise RuntimeError("横线搜索范围为空。")
    darkness = np.sum(255 - gray[top:bottom].astype(np.int32), axis=1)
    return top + int(np.argmax(darkness))


def detect_rectified_geometry(
    image: np.ndarray, region: TableRegion
) -> PageGeometry:
    """Locate columns and 31 ink anchors after photographic dewarping."""

    height, width = image.shape[:2]
    left_source = region.source_bbox[0]
    vertical_rules = [
        min(width - 1, max(0, round(x - left_source)))
        for x in region.source_rules
    ]
    vertical_rules[0] = 0
    vertical_rules[-1] = width - 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    header_bottom = darkest_rule_row(
        gray, round(height * 0.035), round(height * 0.095)
    )

    # The daily body occupies a stable fraction after mapping the outer table
    # borders to a rectangle.  The day-number ink still proves every one of
    # the 31 rows; this boundary only prevents the first statistics row from
    # becoming a false day anchor.
    statistics_top = round(height * 0.780)
    candidates = build_day_column_candidates(
        image, vertical_rules, header_bottom + 2, statistics_top - 2
    )
    anchor_method = "day-column-projection"
    try:
        anchors = choose_31_anchors(
            candidates, header_bottom + 2, statistics_top - 2
        )
    except RuntimeError as day_error:
        candidates = build_multicolumn_support_candidates(
            image, vertical_rules, header_bottom + 2, statistics_top - 2
        )
        try:
            anchors = choose_31_anchors(
                candidates, header_bottom + 2, statistics_top - 2
            )
        except RuntimeError as support_error:
            raise RuntimeError(
                "日列锚点失败，跨列一致性锚点也失败："
                f"日列={day_error}；跨列={support_error}"
            ) from support_error
        anchor_method = "multicolumn-support-projection"
    horizontal_rules = [0, header_bottom, statistics_top, height - 1]
    return PageGeometry(
        page=region.page,
        table_index=region.table_index,
        width=width,
        height=height,
        table_left=0,
        table_top=0,
        table_right=width - 1,
        header_bottom=header_bottom,
        statistics_top=statistics_top,
        vertical_rules=vertical_rules,
        horizontal_rules=horizontal_rules,
        candidate_anchors=candidates,
        row_anchors=anchors,
        anchor_method=anchor_method,
    )


def anchor_row_bounds(
    anchors: list[AnchorCandidate],
    index: int,
    body_top: int,
    body_bottom: int,
) -> tuple[int, int]:
    if index == 0:
        top = body_top
    else:
        top = round((anchors[index - 1].center + anchors[index].center) / 2)
    if index == len(anchors) - 1:
        bottom = body_bottom
    else:
        bottom = round((anchors[index].center + anchors[index + 1].center) / 2)
    return top, bottom


def _fit_statistics_guided_row_curves_legacy(
    image: np.ndarray, geometry: PageGeometry
) -> None:
    """Track a non-crossing row-anchor chain through all 13 columns.

    The printed day column supplies the 1-31 order.  Each month column then
    contributes a new local ink anchor.  Forward and backward passes keep a
    difficult column from propagating its error through the remaining months;
    a regularized fit removes isolated glyph errors while retaining folds.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    centers_x = [
        (geometry.vertical_rules[index] + geometry.vertical_rules[index + 1])
        / 2
        for index in range(EXPECTED_COLUMN_COUNT)
    ]
    base_centers = np.asarray(
        [anchor.center for anchor in geometry.row_anchors], dtype=float
    )
    day_progress = (base_centers - geometry.header_bottom) / max(
        1.0, geometry.statistics_top - geometry.header_bottom
    )
    statistics_samples = fit_statistics_separator_samples(
        gray, page_ink, geometry, centers_x, row_pitch
    )

    statistics_array = np.asarray(statistics_samples, dtype=float)
    coarse_by_day = base_centers[:, None] + day_progress[:, None] * (
        statistics_array[None, :] - geometry.statistics_top
    )
    observations = np.full(
        (EXPECTED_DAY_COUNT, EXPECTED_COLUMN_COUNT), np.nan, dtype=float
    )
    weights = np.zeros_like(observations)
    projections: list[np.ndarray] = []
    column_widths: list[int] = []
    for column in range(EXPECTED_COLUMN_COUNT):
        left = vertical_rule_x(geometry, column, body_midpoint) + 5
        right = vertical_rule_x(geometry, column + 1, body_midpoint) - 5
        column_width = max(1, right - left)
        column_widths.append(column_width)
        expected = coarse_by_day[:, column]
        projection = np.count_nonzero(page_ink[:, left:right], axis=1)
        projections.append(projection)
        for day_index in range(EXPECTED_DAY_COUNT):
            if day_index == 0:
                local_top = geometry.header_bottom + 2
            else:
                local_top = round((expected[day_index - 1] + expected[day_index]) / 2)
            if day_index == EXPECTED_DAY_COUNT - 1:
                local_bottom = round(statistics_samples[column] - 2)
            else:
                local_bottom = round((expected[day_index] + expected[day_index + 1]) / 2)
            local_top = max(0, local_top)
            local_bottom = min(geometry.height, local_bottom)
            if local_bottom <= local_top + 1:
                continue
            values = projection[local_top:local_bottom].astype(float)
            peak = float(np.max(values)) if values.size else 0.0
            total = float(np.sum(values))
            if peak < 2 or total < max(6.0, column_width * 0.055):
                continue
            threshold = max(1.0, peak * 0.16)
            selected = values >= threshold
            if not np.any(selected):
                continue
            y_values = np.arange(local_top, local_bottom, dtype=float)
            selected_weights = values * selected
            centroid = float(
                np.average(y_values, weights=selected_weights)
            )
            observations[day_index, column] = centroid
            weights[day_index, column] = min(
                3.0, 0.5 + peak / max(1.0, column_width * 0.10)
            )

    minimum_chain_gap = max(
        2.0, float(np.median(np.diff(base_centers))) * 0.20
    )

    def observe_anchor_column(
        column: int, predicted: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Attach ordered predictions to local ink inside disjoint row bands."""

        tracked = predicted.astype(float).copy()
        confidence = np.zeros(EXPECTED_DAY_COUNT, dtype=float)
        projection = projections[column]
        column_width = column_widths[column]
        for day_index in range(EXPECTED_DAY_COUNT):
            if day_index == 0:
                local_top = geometry.header_bottom + 2
            else:
                local_top = round(
                    (predicted[day_index - 1] + predicted[day_index]) / 2
                )
            if day_index == EXPECTED_DAY_COUNT - 1:
                local_bottom = round(statistics_samples[column] - 2)
            else:
                local_bottom = round(
                    (predicted[day_index] + predicted[day_index + 1]) / 2
                )
            local_top = max(0, local_top)
            local_bottom = min(geometry.height, local_bottom)
            if local_bottom <= local_top + 1:
                continue
            values = projection[local_top:local_bottom].astype(float)
            peak = float(np.max(values)) if values.size else 0.0
            total = float(np.sum(values))
            if peak < 2 or total < max(6.0, column_width * 0.055):
                continue
            selected = values >= max(1.0, peak * 0.16)
            if not np.any(selected):
                continue
            y_values = np.arange(local_top, local_bottom, dtype=float)
            selected_weights = values * selected
            centroid = float(np.average(y_values, weights=selected_weights))
            evidence = min(
                3.0, 0.5 + peak / max(1.0, column_width * 0.10)
            )
            # Retain a small motion prior so a tall glyph cannot pull the chain
            # all the way to the edge of the neighbouring row's band.
            attachment = min(0.86, 0.55 + evidence * 0.09)
            tracked[day_index] = (
                attachment * centroid + (1.0 - attachment) * predicted[day_index]
            )
            confidence[day_index] = evidence

        for day_index in range(1, EXPECTED_DAY_COUNT):
            tracked[day_index] = max(
                tracked[day_index],
                tracked[day_index - 1] + minimum_chain_gap,
            )
        for day_index in range(EXPECTED_DAY_COUNT - 2, -1, -1):
            tracked[day_index] = min(
                tracked[day_index],
                tracked[day_index + 1] - minimum_chain_gap,
            )
        return tracked, confidence

    # Forward propagation follows the physical reading direction from the
    # proven day-number anchors.  The statistics boundary supplies only the
    # expected inter-column deformation, not an invented row position.
    forward = np.zeros_like(observations)
    forward_confidence = np.zeros_like(weights)
    forward[:, 0] = base_centers
    forward_confidence[:, 0] = 3.0
    for column in range(1, EXPECTED_COLUMN_COUNT):
        predicted = forward[:, column - 1] + (
            coarse_by_day[:, column] - coarse_by_day[:, column - 1]
        )
        forward[:, column], forward_confidence[:, column] = (
            observe_anchor_column(column, predicted)
        )

    # A reverse pass prevents a single weak/dirty month from becoming a
    # permanent accumulated offset.  Its right-edge seed prefers direct ink
    # but safely falls back to the coarse deformation estimate for blank cells.
    backward = np.zeros_like(observations)
    backward_confidence = np.zeros_like(weights)
    right_valid = np.isfinite(observations[:, -1])
    backward[:, -1] = np.where(
        right_valid, observations[:, -1], coarse_by_day[:, -1]
    )
    backward_confidence[:, -1] = np.where(
        right_valid, weights[:, -1], 0.15
    )
    for column in range(EXPECTED_COLUMN_COUNT - 2, -1, -1):
        predicted = backward[:, column + 1] + (
            coarse_by_day[:, column] - coarse_by_day[:, column + 1]
        )
        backward[:, column], backward_confidence[:, column] = (
            observe_anchor_column(column, predicted)
        )

    # Fuse direct, forward and reverse evidence.  Direct ink remains strongest;
    # the two chains mainly rescue large bends and bridge empty calendar cells.
    for day_index in range(EXPECTED_DAY_COUNT):
        for column in range(EXPECTED_COLUMN_COUNT):
            values: list[float] = []
            evidence_weights: list[float] = []
            if np.isfinite(observations[day_index, column]):
                values.append(float(observations[day_index, column]))
                evidence_weights.append(float(weights[day_index, column]))
            if forward_confidence[day_index, column] > 0:
                values.append(float(forward[day_index, column]))
                evidence_weights.append(
                    float(forward_confidence[day_index, column] * 0.65)
                )
            if backward_confidence[day_index, column] > 0:
                values.append(float(backward[day_index, column]))
                evidence_weights.append(
                    float(backward_confidence[day_index, column] * 0.65)
                )
            if values:
                observations[day_index, column] = float(
                    np.average(values, weights=evidence_weights)
                )
                weights[day_index, column] = min(
                    5.0, float(np.sum(evidence_weights))
                )

    count = EXPECTED_COLUMN_COUNT
    second_difference = np.zeros((count - 2, count), dtype=float)
    for index in range(count - 2):
        second_difference[index, index : index + 3] = (1.0, -2.0, 1.0)
    smoothness = second_difference.T @ second_difference
    fitted = np.zeros_like(observations)
    for day_index in range(EXPECTED_DAY_COUNT):
        coarse = coarse_by_day[day_index]
        observed = observations[day_index]
        row_weights = weights[day_index].copy()
        valid = np.isfinite(observed)
        observed_filled = np.where(valid, observed, coarse)
        # Reject a candidate that escaped its date band because a neighbouring
        # row has unusually tall digits.
        if day_index == 0:
            gap_limit = (base_centers[1] - base_centers[0]) * 0.48
        elif day_index == EXPECTED_DAY_COUNT - 1:
            gap_limit = (base_centers[-1] - base_centers[-2]) * 0.48
        else:
            gap_limit = min(
                base_centers[day_index] - base_centers[day_index - 1],
                base_centers[day_index + 1] - base_centers[day_index],
            ) * 0.48
        outlier = np.abs(observed_filled - coarse) > max(3.0, gap_limit)
        row_weights[outlier] = 0.0
        prior_weight = 0.20
        matrix = (
            np.diag(row_weights + prior_weight)
            + 2.8 * smoothness
        )
        target = row_weights * observed_filled + prior_weight * coarse
        fitted[day_index] = np.linalg.solve(matrix, target)

    # Paper deformation changes continuously down the page, whereas glyph
    # centroids vary slightly with the digits themselves.  Smooth only the
    # residual deformation field, then pin it to the ordered day column.
    residual = (fitted - coarse_by_day).astype(np.float32)
    residual = cv2.GaussianBlur(
        residual, (3, 5), sigmaX=0.8, sigmaY=1.0
    ).astype(float)
    residual -= residual[:, :1]
    fitted = coarse_by_day + residual

    # Enforce strict top-to-bottom order in every column.  Normally this is a
    # no-op; it is a safety net for a deep fold or a nearly blank date row.
    minimum_gap = minimum_chain_gap
    for column in range(EXPECTED_COLUMN_COUNT):
        for day_index in range(1, EXPECTED_DAY_COUNT):
            fitted[day_index, column] = max(
                fitted[day_index, column],
                fitted[day_index - 1, column] + minimum_gap,
            )
        for day_index in range(EXPECTED_DAY_COUNT - 2, -1, -1):
            fitted[day_index, column] = min(
                fitted[day_index, column],
                fitted[day_index + 1, column] - minimum_gap,
            )

    geometry.column_center_x = [float(value) for value in centers_x]
    geometry.row_curve_centers = fitted.tolist()
    geometry.statistics_curve = statistics_samples


def _fit_independent_sequence_row_curves_experimental(
    image: np.ndarray, geometry: PageGeometry
) -> None:
    """Build row curves from independently ordered ink anchors in every column.

    The day column proves the 1-31 identities.  Each month column detects its
    own complete top-to-bottom ink sequence and matches that sequence
    monotonically to the day identities.  The statistics boundary is measured
    only for diagnostics and never drives a daily-row position.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    centers_x = [
        (geometry.vertical_rules[index] + geometry.vertical_rules[index + 1])
        / 2
        for index in range(EXPECTED_COLUMN_COUNT)
    ]
    base_centers = np.asarray(
        [anchor.center for anchor in geometry.row_anchors], dtype=float
    )
    ordinary_gaps = np.diff(base_centers)
    ordinary_gaps = ordinary_gaps[
        ordinary_gaps <= np.percentile(ordinary_gaps, 70)
    ]
    row_pitch = float(np.median(ordinary_gaps)) if ordinary_gaps.size else 10.0
    row_pitch = max(4.0, row_pitch)

    # At this stage the scalar table boundary is safer than a raw darkest-row
    # search: the latter can lock onto a rule *inside* the statistics block.
    statistics_samples = [
        float(geometry.statistics_top)
    ] * EXPECTED_COLUMN_COUNT

    observations = np.full(
        (EXPECTED_DAY_COUNT, EXPECTED_COLUMN_COUNT), np.nan, dtype=float
    )
    weights = np.zeros_like(observations)
    observations[:, 0] = base_centers
    weights[:, 0] = 5.0

    def valid_day_options(column: int) -> tuple[int, ...]:
        if column == 0:
            return (31,)
        if column == 2:
            return (28, 29)
        if column in (4, 6, 9, 11):
            return (30,)
        return (31,)

    def select_ordered_subset(
        candidates: list[tuple[float, float]], target_count: int
    ) -> list[tuple[float, float]] | None:
        """Select a rhythm-preserving monotone subset without inventing rows."""

        count = len(candidates)
        if count < target_count:
            return None
        if count == target_count:
            return candidates
        centers = np.asarray([item[0] for item in candidates], dtype=float)
        candidate_gaps = np.diff(centers)
        pitch_like = candidate_gaps[
            (candidate_gaps >= row_pitch * 0.45)
            & (candidate_gaps <= row_pitch * 1.55)
        ]
        scale = (
            float(np.median(pitch_like)) / row_pitch
            if pitch_like.size
            else 1.0
        )
        scale = min(1.30, max(0.70, scale))
        expected = base_centers[:target_count]
        infinity = 1.0e12
        dp = np.full((target_count, count), infinity, dtype=float)
        parent = np.full((target_count, count), -1, dtype=int)
        for current in range(count):
            dp[0, current] = (
                abs(centers[current] - expected[0]) / row_pitch * 0.25
                + current * 0.20
            )
        for day_index in range(1, target_count):
            expected_gap = (
                expected[day_index] - expected[day_index - 1]
            ) * scale
            for current in range(day_index, count):
                for previous in range(day_index - 1, current):
                    actual_gap = centers[current] - centers[previous]
                    gap_cost = (
                        abs(actual_gap - expected_gap) / row_pitch
                    ) ** 2 * 3.0
                    skipped = current - previous - 1
                    value = dp[day_index - 1, previous] + gap_cost + skipped * 0.25
                    if value < dp[day_index, current]:
                        dp[day_index, current] = value
                        parent[day_index, current] = previous
        end_costs = dp[-1].copy()
        for current in range(count):
            end_costs[current] += (count - current - 1) * 0.20
        end = int(np.argmin(end_costs))
        if not np.isfinite(end_costs[end]) or end_costs[end] >= infinity / 2:
            return None
        indices = [end]
        for day_index in range(target_count - 1, 0, -1):
            end = int(parent[day_index, end])
            if end < 0:
                return None
            indices.append(end)
        return [candidates[index] for index in reversed(indices)]

    for column in range(1, EXPECTED_COLUMN_COUNT):
        left = geometry.vertical_rules[column] + 6
        right = geometry.vertical_rules[column + 1] - 6
        column_width = max(1, right - left)
        projection = np.count_nonzero(page_ink[:, left:right], axis=1)
        body_top = max(
            geometry.header_bottom + 2,
            round(base_centers[0] - row_pitch * 0.70),
        )
        body_bottom = min(
            geometry.height,
            round(base_centers[-1] + row_pitch * 1.35),
        )
        options: list[
            tuple[tuple[float, float, int, int], list[tuple[float, float]], int]
        ] = []
        for threshold in range(1, 13):
            bands = active_bands(
                projection[body_top:body_bottom] >= threshold,
                body_top,
            )
            candidates: list[tuple[float, float]] = []
            for top, bottom in bands:
                height = bottom - top + 1
                values = projection[top : bottom + 1].astype(float)
                peak = float(np.max(values)) if values.size else 0.0
                total = float(np.sum(values))
                if not (2 <= height <= row_pitch * 0.88):
                    continue
                # A printed horizontal rule spans most of a month column; a
                # number never does.  Reject the rule before row matching.
                if peak >= column_width * 0.72:
                    continue
                y_values = np.arange(top, bottom + 1, dtype=float)
                centroid = float(
                    np.average(y_values, weights=values)
                    if total > 0
                    else (top + bottom) / 2
                )
                evidence = min(
                    4.0,
                    0.5
                    + peak / max(1.0, column_width * 0.10)
                    + total / max(1.0, column_width * row_pitch * 0.18),
                )
                candidates.append((centroid, evidence))

            for target_count in valid_day_options(column):
                selected = select_ordered_subset(candidates, target_count)
                if selected is None:
                    continue
                selected_centers = np.asarray(
                    [item[0] for item in selected], dtype=float
                )
                expected = base_centers[:target_count]
                degree = 2 if target_count >= 5 else 1
                coefficients = np.polyfit(expected, selected_centers, degree)
                residual = float(
                    np.sqrt(
                        np.mean(
                            (
                                selected_centers
                                - np.polyval(coefficients, expected)
                            )
                            ** 2
                        )
                    )
                    / row_pitch
                )
                options.append(
                    (
                        (
                            abs(len(candidates) - target_count),
                            residual,
                            threshold,
                            -target_count,
                        ),
                        selected,
                        target_count,
                    )
                )

        if options:
            _, selected, target_count = min(options, key=lambda item: item[0])
            for day_index, (center, evidence) in enumerate(selected):
                observations[day_index, column] = center
                weights[day_index, column] = evidence
            continue

        # Conservative fallback: estimate one column-local offset from ink and
        # recenter every day independently.  It does not use neighbouring
        # columns or the statistics line, so any failure remains local.
        target_count = max(valid_day_options(column))
        shift_candidates = np.linspace(-row_pitch * 1.5, row_pitch * 1.5, 61)
        shift_scores: list[float] = []
        half_window = max(3, round(row_pitch * 0.28))
        for shift in shift_candidates:
            score = 0.0
            for expected in base_centers[:target_count] + shift:
                top = max(0, round(expected) - half_window)
                bottom = min(geometry.height, round(expected) + half_window + 1)
                score += float(np.max(projection[top:bottom]))
            shift_scores.append(score)
        shift = float(shift_candidates[int(np.argmax(shift_scores))])
        expected_centers = base_centers + shift
        for day_index in range(target_count):
            if day_index == 0:
                top = geometry.header_bottom + 2
            else:
                top = round(
                    (expected_centers[day_index - 1] + expected_centers[day_index])
                    / 2
                )
            if day_index == EXPECTED_DAY_COUNT - 1:
                bottom = body_bottom
            else:
                bottom = round(
                    (expected_centers[day_index] + expected_centers[day_index + 1])
                    / 2
                )
            values = projection[max(0, top) : min(geometry.height, bottom)].astype(float)
            if values.size == 0 or float(np.max(values)) < 2:
                continue
            selected_rows = values >= max(1.0, float(np.max(values)) * 0.16)
            selected_weights = values * selected_rows
            if float(np.sum(selected_weights)) <= 0:
                continue
            y_values = np.arange(max(0, top), min(geometry.height, bottom), dtype=float)
            observations[day_index, column] = float(
                np.average(y_values, weights=selected_weights)
            )
            weights[day_index, column] = 1.0

    displacement = observations - base_centers[:, None]
    filled_displacement = np.zeros_like(displacement)
    day_indices = np.arange(EXPECTED_DAY_COUNT, dtype=float)
    for column in range(EXPECTED_COLUMN_COUNT):
        valid = np.isfinite(displacement[:, column])
        if np.count_nonzero(valid) >= 2:
            filled_displacement[:, column] = np.interp(
                day_indices,
                day_indices[valid],
                displacement[valid, column],
            )
        elif np.count_nonzero(valid) == 1:
            filled_displacement[:, column] = float(displacement[valid, column][0])

    # Smooth only the independently observed displacement field.  Direct ink
    # keeps most of the weight; smoothing bridges calendar blanks and removes
    # digit-shape centroid jitter without propagating a statistics-line error.
    smooth_displacement = cv2.GaussianBlur(
        filled_displacement.astype(np.float32),
        (3, 5),
        sigmaX=0.65,
        sigmaY=0.9,
    ).astype(float)
    fitted_displacement = smooth_displacement.copy()
    valid = np.isfinite(displacement)
    fitted_displacement[valid] = (
        displacement[valid] * 0.72 + smooth_displacement[valid] * 0.28
    )
    fitted_displacement[:, 0] = 0.0
    fitted = base_centers[:, None] + fitted_displacement

    minimum_gap = max(2.0, row_pitch * 0.20)
    for column in range(EXPECTED_COLUMN_COUNT):
        for day_index in range(1, EXPECTED_DAY_COUNT):
            fitted[day_index, column] = max(
                fitted[day_index, column],
                fitted[day_index - 1, column] + minimum_gap,
            )
        for day_index in range(EXPECTED_DAY_COUNT - 2, -1, -1):
            fitted[day_index, column] = min(
                fitted[day_index, column],
                fitted[day_index + 1, column] - minimum_gap,
            )

    geometry.column_center_x = [float(value) for value in centers_x]
    geometry.row_curve_centers = fitted.tolist()
    geometry.statistics_curve = statistics_samples


def fit_dynamic_vertical_rule_models(
    image: np.ndarray, geometry: PageGeometry, row_pitch: float
) -> None:
    """Fit every photographed table rule as x(y), not as a fixed x value."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_column_width = float(np.median(np.diff(geometry.vertical_rules)))
    search_radius = max(6, round(median_column_width * 0.105))
    half_band = max(4, round(row_pitch * 0.24))
    top = max(1, geometry.table_top)
    bottom = min(geometry.height - 2, geometry.statistics_top)
    sample_y = np.linspace(top, bottom, 90).astype(int)
    # A curved rule appears in a probabilistic Hough transform as several
    # short, near-vertical segments.  Their endpoints seed the later robust
    # x(y) fit; they are not mistaken for one global straight line.
    edges = cv2.Canny(gray[top:bottom], 55, 150, apertureSize=3)
    hough_lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=max(12, round(row_pitch * 0.8)),
        minLineLength=max(12, round(row_pitch * 1.6)),
        maxLineGap=max(5, round(row_pitch * 0.75)),
    )
    hough_points: list[list[tuple[float, float, float]]] = [
        [] for _ in geometry.vertical_rules
    ]
    if hough_lines is not None:
        for line in np.asarray(hough_lines).reshape(-1, 4):
            x1, local_y1, x2, local_y2 = map(float, line)
            y1 = local_y1 + top
            y2 = local_y2 + top
            vertical_span = abs(y2 - y1)
            if vertical_span < row_pitch * 1.4:
                continue
            if abs(x2 - x1) > max(3.0, vertical_span * 0.24):
                continue
            midpoint_x = (x1 + x2) / 2
            rule_index = int(
                np.argmin(
                    np.abs(
                        np.asarray(geometry.vertical_rules, dtype=float)
                        - midpoint_x
                    )
                )
            )
            if (
                abs(midpoint_x - geometry.vertical_rules[rule_index])
                > search_radius * 1.25
            ):
                continue
            segment_weight = min(255.0 * vertical_span, 255.0 * row_pitch * 5)
            hough_points[rule_index].extend(
                [
                    (y1, x1, segment_weight),
                    ((y1 + y2) / 2, midpoint_x, segment_weight),
                    (y2, x2, segment_weight),
                ]
            )
    models: list[list[float]] = []
    for rule_index, nominal_x in enumerate(geometry.vertical_rules):
        observed_y: list[float] = []
        observed_x: list[float] = []
        observed_weight: list[float] = []
        corridor_left = max(0, nominal_x - search_radius)
        corridor_right = min(geometry.width, nominal_x + search_radius + 1)
        for center_y in sample_y:
            band_top = max(0, center_y - half_band)
            band_bottom = min(geometry.height, center_y + half_band + 1)
            band = 255 - gray[band_top:band_bottom, corridor_left:corridor_right]
            if band.size == 0:
                continue
            darkness = np.sum(band.astype(np.float32), axis=0)
            xs = np.arange(corridor_left, corridor_right, dtype=float)
            # The rule must remain close to its nominal component center.  A
            # mild penalty prevents a repeated digit stroke from winning one
            # local band while still permitting genuine slant/curvature.
            penalized = darkness - np.abs(xs - nominal_x) * (
                max(1.0, float(np.percentile(darkness, 80))) * 0.018
            )
            local_index = int(np.argmax(penalized))
            peak = float(darkness[local_index])
            if peak < (band_bottom - band_top) * 255 * 0.28:
                continue
            observed_y.append(float(center_y))
            observed_x.append(float(xs[local_index]))
            observed_weight.append(peak)

        for hough_y, hough_x, hough_weight in hough_points[rule_index]:
            observed_y.append(float(hough_y))
            observed_x.append(float(hough_x))
            observed_weight.append(float(hough_weight))

        if len(observed_y) < 12:
            models.append([0.0, 0.0, float(nominal_x)])
            continue
        ys = np.asarray(observed_y, dtype=float)
        xs = np.asarray(observed_x, dtype=float)
        weights = np.sqrt(np.asarray(observed_weight, dtype=float))
        inliers = np.ones(len(ys), dtype=bool)
        coefficients = np.asarray([0.0, 0.0, float(nominal_x)])
        for _ in range(5):
            degree = 2 if int(np.count_nonzero(inliers)) >= 18 else 1
            fitted_coefficients = np.polyfit(
                ys[inliers], xs[inliers], degree, w=weights[inliers]
            )
            if degree == 1:
                fitted_coefficients = np.asarray(
                    [0.0, fitted_coefficients[0], fitted_coefficients[1]]
                )
            predicted = np.polyval(fitted_coefficients, ys)
            residual = np.abs(xs - predicted)
            updated = residual <= max(2.0, search_radius * 0.22)
            coefficients = fitted_coefficients
            if int(np.count_nonzero(updated)) < 10 or np.array_equal(updated, inliers):
                break
            inliers = updated
        check_y = np.linspace(top, bottom, 25)
        check_x = np.polyval(coefficients, check_y)
        if (
            not np.all(np.isfinite(check_x))
            or float(np.max(np.abs(check_x - nominal_x))) > search_radius * 1.05
        ):
            coefficients = np.asarray([0.0, 0.0, float(nominal_x)])
        models.append([float(value) for value in coefficients])
    geometry.vertical_rule_models = models


def vertical_rule_x(geometry: PageGeometry, rule_index: int, y: float) -> float:
    if (
        geometry.vertical_rule_models
        and rule_index < len(geometry.vertical_rule_models)
    ):
        return float(np.polyval(geometry.vertical_rule_models[rule_index], y))
    return float(geometry.vertical_rules[rule_index])


def fit_horizontal_boundary_samples(
    gray: np.ndarray,
    geometry: PageGeometry,
    centers_x: list[float],
    nominal_y: float,
    search_radius: int,
) -> list[float]:
    """Track one photographed horizontal rule as y(x) from local evidence.

    A bowed/folded rule is represented by short near-horizontal Hough segments
    plus a local darkness peak in every logical column.  The returned samples
    preserve column order and are later interpolated with PCHIP; no rectangular
    boundary is imposed on the daily body.
    """

    height, width = gray.shape[:2]
    strip_top = max(0, round(nominal_y) - search_radius)
    strip_bottom = min(height, round(nominal_y) + search_radius + 1)
    if strip_bottom <= strip_top + 2:
        return [float(nominal_y)] * EXPECTED_COLUMN_COUNT
    median_width = float(np.median(np.diff(geometry.vertical_rules)))
    edges = cv2.Canny(gray[strip_top:strip_bottom], 45, 145, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=max(10, round(median_width * 0.13)),
        minLineLength=max(12, round(median_width * 0.34)),
        maxLineGap=max(5, round(median_width * 0.12)),
    )
    hough_y: list[list[float]] = [[] for _ in range(EXPECTED_COLUMN_COUNT)]
    if lines is not None:
        for x1, local_y1, x2, local_y2 in np.asarray(lines).reshape(-1, 4):
            x1, x2 = float(x1), float(x2)
            y1, y2 = float(local_y1 + strip_top), float(local_y2 + strip_top)
            span = abs(x2 - x1)
            if span < median_width * 0.30:
                continue
            if abs(y2 - y1) > max(4.0, span * 0.18):
                continue
            low_x, high_x = sorted((x1, x2))
            for column, center_x in enumerate(centers_x):
                if low_x - 3 <= center_x <= high_x + 3:
                    ratio = (center_x - x1) / (x2 - x1) if x2 != x1 else 0.5
                    hough_y[column].append(y1 + ratio * (y2 - y1))

    samples: list[float] = []
    for column, center_x in enumerate(centers_x):
        reference = (
            float(np.median(hough_y[column]))
            if hough_y[column]
            else float(nominal_y)
        )
        local_top = max(strip_top, round(reference) - search_radius)
        local_bottom = min(strip_bottom, round(reference) + search_radius + 1)
        half_width = max(8, round(median_width * 0.42))
        left = max(0, round(center_x) - half_width)
        right = min(width, round(center_x) + half_width + 1)
        candidates = list(range(local_top, local_bottom))
        if not candidates or right <= left:
            samples.append(float(reference))
            continue
        scores: list[float] = []
        for y in candidates:
            row = 255 - gray[y, left:right].astype(np.int32)
            darkness = float(np.sum(row))
            continuity = float(np.count_nonzero(row >= 70))
            prior = abs(y - reference) * max(1.0, median_width) * 255 * 0.05
            scores.append(darkness + continuity * 255 * 1.4 - prior)
        samples.append(float(candidates[int(np.argmax(scores))]))

    values = np.asarray(samples, dtype=float)
    local_median = cv2.medianBlur(values.astype(np.float32)[None, :], 3).reshape(-1)
    bad = np.abs(values - local_median) > max(4.0, search_radius * 0.38)
    values[bad] = local_median[bad]
    slope_cap = max(3.0, search_radius * 0.35)
    for _ in range(3):
        for column in range(1, EXPECTED_COLUMN_COUNT):
            values[column] = np.clip(
                values[column],
                values[column - 1] - slope_cap,
                values[column - 1] + slope_cap,
            )
        for column in range(EXPECTED_COLUMN_COUNT - 2, -1, -1):
            values[column] = np.clip(
                values[column],
                values[column + 1] - slope_cap,
                values[column + 1] + slope_cap,
            )
    return values.astype(float).tolist()


def boundary_values_at_rules(
    geometry: PageGeometry, values: list[float], fallback: float
) -> list[float]:
    if values and geometry.column_center_x:
        return curve_values_at_rules(geometry, values)
    return [float(fallback)] * len(geometry.vertical_rules)


def fit_statistics_separator_samples(
    gray: np.ndarray,
    page_ink: np.ndarray,
    geometry: PageGeometry,
    centers_x: list[float],
    row_pitch: float,
    lower_bounds: list[float] | None = None,
) -> list[float]:
    """Split the statistics section before any month-row anchoring."""

    search_radius = max(9, round(geometry.height * 0.024), round(row_pitch * 1.15))
    strip_top = max(0, geometry.statistics_top - search_radius)
    strip_bottom = min(geometry.height, geometry.statistics_top + search_radius + 1)
    strip_ink = page_ink[strip_top:strip_bottom]
    median_width = float(np.median(np.diff(geometry.vertical_rules)))
    horizontal = cv2.morphologyEx(
        strip_ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(11, round(median_width * 0.34)), 1)
        ),
    )

    # Near-horizontal Hough segments provide a second, independent boundary
    # observation when a bowed separator occupies several local slopes.
    edges = cv2.Canny(gray[strip_top:strip_bottom], 50, 145, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360,
        threshold=max(12, round(median_width * 0.16)),
        minLineLength=max(14, round(median_width * 0.42)),
        maxLineGap=max(5, round(median_width * 0.12)),
    )
    hough_y_by_column: list[list[float]] = [
        [] for _ in range(EXPECTED_COLUMN_COUNT)
    ]
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            x1 = float(x1)
            x2 = float(x2)
            y1 = float(y1 + strip_top)
            y2 = float(y2 + strip_top)
            horizontal_span = abs(x2 - x1)
            if horizontal_span < median_width * 0.35:
                continue
            if abs(y2 - y1) > max(4.0, horizontal_span * 0.16):
                continue
            low_x, high_x = sorted((x1, x2))
            for column, center_x in enumerate(centers_x):
                if low_x - 3 <= center_x <= high_x + 3:
                    ratio = (center_x - x1) / (x2 - x1) if x2 != x1 else 0.5
                    hough_y_by_column[column].append(y1 + ratio * (y2 - y1))

    samples: list[float] = []
    for column in range(EXPECTED_COLUMN_COUNT):
        lower_bound = (
            float(lower_bounds[column])
            if lower_bounds is not None and column < len(lower_bounds)
            else float(strip_top)
        )
        # Several horizontal rules exist in the statistics block.  The daily
        # body's lower edge is the *first* physical rule below the last valid
        # daily glyph, not the darkest rule in the search strip.  Using the
        # median/all-row darkness here used to select the rule below the
        # maximum-value row and made a day-31 crop contain both values.
        eligible_hough = sorted(
            value
            for value in hough_y_by_column[column]
            if value >= lower_bound - 1.5
        )
        reference_y = (
            float(eligible_hough[0])
            if eligible_hough
            else float(geometry.statistics_top)
        )
        search_top = max(strip_top, round(reference_y) - search_radius)
        search_bottom = min(strip_bottom, round(reference_y) + search_radius + 1)
        scores: list[float] = []
        candidate_y = list(range(search_top, search_bottom))
        strong_rule_rows: list[int] = []
        for y in candidate_y:
            left = round(vertical_rule_x(geometry, column, y)) + 3
            right = round(vertical_rule_x(geometry, column + 1, y)) - 3
            left = max(0, left)
            right = min(geometry.width, right)
            if right <= left:
                scores.append(0.0)
                continue
            local_y = y - strip_top
            line_support = float(np.count_nonzero(horizontal[local_y, left:right]))
            ink_support = float(np.count_nonzero(page_ink[y, left:right]))
            width = max(1, right - left)
            darkness = float(
                np.sum(255 - gray[y, left:right].astype(np.int32))
            )
            hough_bonus = (
                max(0.0, 1.0 - abs(y - reference_y) / 3.0)
                * (right - left)
                * 255
                * 0.35
                if hough_y_by_column[column]
                else 0.0
            )
            scores.append(line_support * 255 * 2.2 + darkness + hough_bonus)
            if (
                y >= lower_bound - 1.0
                and (
                    line_support / width >= 0.28
                    or ink_support / width >= 0.68
                )
            ):
                strong_rule_rows.append(y)
        if not scores:
            samples.append(float(geometry.statistics_top))
        elif strong_rule_rows:
            # Select the first consecutive rule band, then the strongest pixel
            # inside that band.  This retains a locally slanted/bowed divider
            # while preventing any later statistics-row rule from winning.
            first_band = [strong_rule_rows[0]]
            for y in strong_rule_rows[1:]:
                if y <= first_band[-1] + 1:
                    first_band.append(y)
                else:
                    break
            samples.append(
                float(
                    max(
                        first_band,
                        key=lambda y: scores[y - search_top],
                    )
                )
            )
        else:
            # With no sufficiently long physical rule, keep the estimate near
            # the initial table boundary instead of drifting to darker digits.
            adjusted = [
                score
                - abs(y - geometry.statistics_top)
                * max(1.0, median_width)
                * 255.0
                * 0.22
                for y, score in zip(candidate_y, scores)
            ]
            samples.append(float(candidate_y[int(np.argmax(adjusted))]))
    return samples


def detect_month_decimal_track(
    page_ink: np.ndarray,
    geometry: PageGeometry,
    column: int,
    row_pitch: float,
    body_bottom: int,
) -> tuple[tuple[float, float] | None, list[dict[str, float]]]:
    """Find the repeated decimal-point track inside one flow column.

    This follows the proven water-level anchor idea in
    ``flood_hydro_factor_pdf_ex2.py``: dot-sized components are first found
    independently, then only components agreeing with one slightly slanted
    x(y) track survive.  Requiring a repeated track prevents isolated dirt or
    a broken digit stroke from becoming a date anchor.
    """

    body_top = (
        max(
            geometry.header_bottom + 2,
            round(geometry.header_curve[column]) + 2,
        )
        if geometry.header_curve and column < len(geometry.header_curve)
        else geometry.header_bottom + 2
    )
    boundary_samples = [float(body_top), float(body_bottom - 1)]
    left = max(
        0,
        int(
            np.floor(
                min(vertical_rule_x(geometry, column, y) for y in boundary_samples)
            )
        ),
    )
    right = min(
        geometry.width,
        int(
            np.ceil(
                max(
                    vertical_rule_x(geometry, column + 1, y)
                    for y in boundary_samples
                )
            )
        )
        + 1,
    )
    width = max(
        1.0,
        vertical_rule_x(geometry, column + 1, (body_top + body_bottom) / 2)
        - vertical_rule_x(geometry, column, (body_top + body_bottom) / 2),
    )
    source = page_ink[body_top:body_bottom, left:right].copy()
    for local_y, absolute_y in enumerate(range(body_top, body_bottom)):
        allowed_left = round(vertical_rule_x(geometry, column, absolute_y)) + 5
        allowed_right = (
            round(vertical_rule_x(geometry, column + 1, absolute_y)) - 5
        )
        source[local_y, : max(0, allowed_left - left)] = 0
        source[local_y, max(0, allowed_right - left) :] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(source, 8)
    candidates: list[dict[str, float]] = []
    maximum_side = max(4, round(row_pitch * 0.27))
    maximum_area = max(18, round(row_pitch * row_pitch * 0.055))
    for component_index in range(1, count):
        x, y, component_width, component_height, area = map(
            int, stats[component_index]
        )
        center_x, center_y = map(float, centroids[component_index])
        fill_ratio = area / max(1, component_width * component_height)
        if not (
            2 <= component_width <= maximum_side
            and 1 <= component_height <= maximum_side
            and 3 <= area <= maximum_area
            and 0.32 <= fill_ratio <= 1.0
            and component_width <= component_height * 2.4
            and component_height <= component_width * 2.4
        ):
            continue
        absolute_y = center_y + body_top
        dynamic_left = vertical_rule_x(geometry, column, absolute_y) + 5
        dynamic_right = vertical_rule_x(geometry, column + 1, absolute_y) - 5
        dynamic_width = max(1.0, dynamic_right - dynamic_left)
        absolute_x = center_x + left
        relative_x = (absolute_x - dynamic_left) / dynamic_width
        if not 0.18 <= relative_x <= 0.88:
            continue
        candidates.append(
            {
                "x": absolute_x,
                "y": absolute_y,
                "area": float(area),
                "fill": float(fill_ratio),
            }
        )
    if len(candidates) < 2:
        return None, []

    # Search a narrow family of vertical/slightly slanted tracks.  Counting
    # distinct y bands, rather than raw components, prevents several fragments
    # from the same glyph winning the vote.
    tolerance = max(1.5, row_pitch * 0.095)
    slope_limit = max(0.012, width * 0.075 / max(1.0, body_bottom - body_top))
    best_score = -1.0
    best_inliers: list[dict[str, float]] = []
    y_origin = float(body_top)
    slopes = np.linspace(-slope_limit, slope_limit, 51)
    for slope in slopes:
        intercept_values = [
            item["x"] - slope * (item["y"] - y_origin)
            for item in candidates
        ]
        for seed in intercept_values:
            inliers = [
                item
                for item, intercept in zip(candidates, intercept_values)
                if abs(intercept - seed) <= tolerance
            ]
            ordered = sorted(inliers, key=lambda item: item["y"])
            distinct: list[dict[str, float]] = []
            for item in ordered:
                if (
                    not distinct
                    or item["y"] - distinct[-1]["y"] >= row_pitch * 0.42
                ):
                    distinct.append(item)
                elif item["fill"] * item["area"] > (
                    distinct[-1]["fill"] * distinct[-1]["area"]
                ):
                    distinct[-1] = item
            score = sum(1.0 + item["fill"] * 0.22 for item in distinct)
            if score > best_score:
                best_score = score
                best_inliers = distinct
    if len(best_inliers) < 2:
        return None, []

    y_values = np.asarray([item["y"] for item in best_inliers], dtype=float)
    x_values = np.asarray([item["x"] for item in best_inliers], dtype=float)
    slope, absolute_intercept = np.polyfit(y_values, x_values, 1)
    residuals = np.abs(x_values - (slope * y_values + absolute_intercept))
    keep = residuals <= tolerance
    if int(np.count_nonzero(keep)) < 2:
        return None, []
    slope, absolute_intercept = np.polyfit(y_values[keep], x_values[keep], 1)
    points = [
        item for item, retained in zip(best_inliers, keep.tolist()) if retained
    ]
    return (float(slope), float(absolute_intercept)), points


def align_observed_sequence_to_days(
    observations: list[dict[str, float]],
    base_centers: np.ndarray,
    preliminary: np.ndarray,
    printed_days: int,
    row_pitch: float,
    offset_range: tuple[float, float],
) -> tuple[dict[int, dict[str, float]], float] | None:
    """Align a sparse observed sequence against the exact day-column rhythm.

    Some source tables intentionally insert an extra vertical gap after every
    five dates.  Therefore a 48-pixel gap is not necessarily a missing day.
    Dynamic programming compares every observed gap with the corresponding
    *actual day-column anchor gap*, preserving such layout rhythms and making
    skipped decimal points local rather than cumulative.
    """

    ordered = sorted(observations, key=lambda item: item["y"])
    observation_count = len(ordered)
    if observation_count < 2 or observation_count > printed_days:
        return None
    y_values = np.asarray([item["y"] for item in ordered], dtype=float)
    infinity = float("inf")
    costs = np.full((observation_count, printed_days), infinity, dtype=float)
    previous = np.full((observation_count, printed_days), -1, dtype=int)
    offset_min, offset_max = offset_range

    def offset_penalty(value: float) -> float:
        if value < offset_min:
            return (offset_min - value) / row_pitch
        if value > offset_max:
            return (value - offset_max) / row_pitch
        return 0.0

    for day_index in range(printed_days):
        residual = y_values[0] - preliminary[day_index]
        costs[0, day_index] = (
            day_index * 0.24 + offset_penalty(residual) * 2.2
        )

    for observation_index in range(1, observation_count):
        observed_gap = y_values[observation_index] - y_values[observation_index - 1]
        for day_index in range(observation_index, printed_days):
            residual = y_values[observation_index] - preliminary[day_index]
            point_cost = offset_penalty(residual) * 1.25
            for prior_day in range(observation_index - 1, day_index):
                prior_cost = costs[observation_index - 1, prior_day]
                if not np.isfinite(prior_cost):
                    continue
                expected_gap = base_centers[day_index] - base_centers[prior_day]
                gap_error = abs(observed_gap - expected_gap) / row_pitch
                skipped_days = day_index - prior_day - 1
                candidate = (
                    prior_cost
                    + gap_error * 2.35
                    + skipped_days * 0.27
                    + point_cost
                )
                if candidate < costs[observation_index, day_index]:
                    costs[observation_index, day_index] = candidate
                    previous[observation_index, day_index] = prior_day

    last_costs = costs[-1].copy()
    for day_index in range(printed_days):
        last_costs[day_index] += (printed_days - day_index - 1) * 0.20
    last_day = int(np.argmin(last_costs))
    if not np.isfinite(last_costs[last_day]):
        return None
    path = [last_day]
    for observation_index in range(observation_count - 1, 0, -1):
        last_day = int(previous[observation_index, last_day])
        if last_day < 0:
            return None
        path.append(last_day)
    path.reverse()

    # Reject a mathematically possible but geometrically implausible path.
    if observation_count > 1:
        transition_errors = []
        for index in range(1, observation_count):
            observed_gap = y_values[index] - y_values[index - 1]
            expected_gap = base_centers[path[index]] - base_centers[path[index - 1]]
            date_span = max(1, path[index] - path[index - 1])
            transition_errors.append(
                abs(observed_gap - expected_gap) / date_span
            )
        if float(np.median(transition_errors)) > row_pitch * 0.34:
            return None

    mapping = {
        int(day_index): item for day_index, item in zip(path, ordered)
    }
    raw_offset = float(
        np.median(
            [
                item["y"] - preliminary[day_index]
                for day_index, item in mapping.items()
            ]
        )
    )
    return mapping, raw_offset


def detect_integer_token_centers_left_of_decimal(
    page_ink: np.ndarray,
    geometry: PageGeometry,
    column: int,
    decimal_model: tuple[float, float],
    row_pitch: float,
    body_bottom: int,
) -> list[dict[str, float]]:
    """Find integer/glyph rows strictly left of the fitted decimal track."""

    body_top = geometry.header_bottom + 2
    boundary_samples = [float(body_top), float(body_bottom - 1)]
    left = max(
        0,
        int(
            np.floor(
                min(vertical_rule_x(geometry, column, y) for y in boundary_samples)
            )
        ),
    )
    right = min(
        geometry.width,
        int(
            np.ceil(
                max(
                    vertical_rule_x(geometry, column + 1, y)
                    for y in boundary_samples
                )
            )
        )
        + 1,
    )
    masked = np.zeros((body_bottom - body_top, right - left), dtype=np.uint8)
    slope, intercept = decimal_model
    for local_y, absolute_y in enumerate(range(body_top, body_bottom)):
        decimal_x = round(slope * absolute_y + intercept)
        dynamic_left = round(vertical_rule_x(geometry, column, absolute_y)) + 6
        dynamic_right = (
            round(vertical_rule_x(geometry, column + 1, absolute_y)) - 6
        )
        start = min(dynamic_right, max(left, dynamic_left))
        limit = min(dynamic_right, right, max(start, decimal_x - 2))
        if limit > start:
            masked[local_y, start - left : limit - left] = page_ink[
                absolute_y, start:limit
            ]
    projection = np.count_nonzero(masked, axis=1).astype(float)
    active = (projection >= 1).astype(np.uint8)[:, None]
    active = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
    ).reshape(-1).astype(bool)
    bands = active_bands(active, body_top)
    tokens: list[dict[str, float]] = []
    for top, bottom in bands:
        height = bottom - top + 1
        values = projection[top - body_top : bottom - body_top + 1]
        total = float(np.sum(values))
        if not (2 <= height <= row_pitch * 0.88 and total >= 5):
            continue
        ys = np.arange(top, bottom + 1, dtype=float)
        center_y = float(np.average(ys, weights=values))
        tokens.append(
            {
                "y": center_y,
                "top": float(top),
                "bottom": float(bottom),
                "ink": total,
            }
        )
    return tokens


def inspect_decimal_slot_occupancy(
    page_ink: np.ndarray,
    geometry: PageGeometry,
    column: int,
    decimal_model: tuple[float, float],
    center_y: float,
    row_pitch: float,
) -> dict[str, float | bool | str]:
    """Classify 119 versus 11.9 from the fitted slot and ink dispersion."""

    top = max(0, round(center_y - row_pitch * 0.40))
    bottom = min(geometry.height, round(center_y + row_pitch * 0.43) + 1)
    if bottom <= top:
        return {
            "layout": "ambiguous",
            "slot_ink": 0.0,
            "left_ink": 0.0,
            "right_ink": 0.0,
            "dispersion": 0.0,
        }
    decimal_x = decimal_model[0] * center_y + decimal_model[1]
    left = max(0, round(vertical_rule_x(geometry, column, center_y)) + 6)
    right = min(
        geometry.width,
        round(vertical_rule_x(geometry, column + 1, center_y)) - 6,
    )
    slot_half_width = max(2, round((right - left) * 0.018))
    slot_left = max(left, round(decimal_x) - slot_half_width)
    slot_right = min(right, round(decimal_x) + slot_half_width + 1)
    left_end = max(left, slot_left - 1)
    right_start = min(right, slot_right + 1)
    left_ink = float(np.count_nonzero(page_ink[top:bottom, left:left_end]))
    slot_ink = float(
        np.count_nonzero(page_ink[top:bottom, slot_left:slot_right])
    )
    right_ink = float(
        np.count_nonzero(page_ink[top:bottom, right_start:right])
    )
    ys, xs = np.nonzero(page_ink[top:bottom, left:right])
    del ys
    dispersion = (
        float(np.std(xs) / max(1.0, right - left)) if xs.size else 0.0
    )
    right_ratio = right_ink / max(1.0, left_ink + slot_ink + right_ink)
    slot_ratio = slot_ink / max(1.0, left_ink + slot_ink + right_ink)
    has_fraction_side = right_ink >= max(4.0, left_ink * 0.055)
    has_decimal_slot = slot_ink >= 2 and slot_ratio >= 0.012
    integer_left_only = left_ink >= 5 and right_ratio <= 0.025 and slot_ink <= 2
    if has_decimal_slot and has_fraction_side:
        layout = "decimal-slot-and-fraction"
    elif has_fraction_side and dispersion >= 0.115:
        layout = "faint-decimal-dispersed-fraction"
    elif integer_left_only:
        layout = "integer-left-only"
    else:
        layout = "ambiguous"
    return {
        "layout": layout,
        "slot_ink": slot_ink,
        "left_ink": left_ink,
        "right_ink": right_ink,
        "right_ink_ratio": float(right_ratio),
        "slot_ink_ratio": float(slot_ratio),
        "dispersion": dispersion,
    }


def apply_month_decimal_anchors(
    image: np.ndarray,
    page_ink: np.ndarray,
    geometry: PageGeometry,
    base_centers: np.ndarray,
    fitted: np.ndarray,
    row_pitch: float,
    body_bottom: int,
    statistics_samples: list[float],
) -> np.ndarray:
    """Strengthen every month with table-consensus dots and left tokens."""

    del image  # Kept in the signature for future grayscale diagnostics.
    diagnostics: list[list[dict[str, Any]]] = [
        [] for _ in range(EXPECTED_COLUMN_COUNT)
    ]
    virtual_diagnostics: list[dict[str, Any]] = []

    def maximum_printed_day(column: int) -> int:
        if column == 2:
            return 29
        if column in (4, 6, 9, 11):
            return 30
        return 31

    def set_calendar_virtual_anchors(column: int, printed_days: int) -> None:
        for day_index in range(printed_days, EXPECTED_DAY_COUNT):
            layout_gap = base_centers[day_index] - base_centers[day_index - 1]
            fitted[day_index, column] = (
                fitted[day_index - 1, column] + layout_gap
            )
            virtual_diagnostics.append(
                {
                    "month": column,
                    "day": day_index + 1,
                    "row_center_y": float(fitted[day_index, column]),
                    "source": "calendar-layout-extrapolation",
                }
            )

    body_midpoint = (geometry.header_bottom + 2 + body_bottom) / 2
    body_span = max(1.0, body_bottom - geometry.header_bottom - 2)
    raw_tracks: dict[int, dict[str, Any]] = {}
    for column in range(1, EXPECTED_COLUMN_COUNT):
        column_body_bottom = min(
            body_bottom, round(statistics_samples[column]) - 2
        )
        model, points = detect_month_decimal_track(
            page_ink, geometry, column, row_pitch, column_body_bottom
        )
        if model is None or len(points) < 2:
            continue
        left = vertical_rule_x(geometry, column, body_midpoint) + 5
        right = vertical_rule_x(geometry, column + 1, body_midpoint) - 5
        width = max(1.0, right - left)
        midpoint_x = model[0] * body_midpoint + model[1]
        raw_tracks[column] = {
            "model": model,
            "points": points,
            "ratio": (midpoint_x - left) / width,
            "normalized_slope": model[0] * body_span / width,
            "weight": float(len(points)),
            "body_bottom": int(column_body_bottom),
        }
    if not raw_tracks:
        for column in range(1, EXPECTED_COLUMN_COUNT):
            set_calendar_virtual_anchors(
                column, maximum_printed_day(column)
            )
        geometry.month_decimal_anchors = diagnostics
        geometry.virtual_month_anchors = virtual_diagnostics
        return fitted

    def weighted_median(values: list[float], weights: list[float]) -> float:
        order = np.argsort(np.asarray(values, dtype=float))
        sorted_values = np.asarray(values, dtype=float)[order]
        sorted_weights = np.asarray(weights, dtype=float)[order]
        cutoff = float(np.sum(sorted_weights)) / 2
        index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff))
        return float(sorted_values[min(index, len(sorted_values) - 1)])

    # Decimal alignment x can legitimately move when one month contains much
    # larger values than another.  Do not force a single x ratio.  A track is
    # trusted when it covers a meaningful portion of that month and its y
    # sequence matches the exact printed day-column rhythm.
    accepted_tracks: dict[int, dict[str, Any]] = {}
    validated_tracks: dict[int, dict[str, Any]] = {}
    dot_alignments: dict[int, dict[int, dict[str, float]]] = {}
    raw_dot_offsets: list[float] = []
    for column, item in raw_tracks.items():
        printed_days = maximum_printed_day(column)
        aligned = align_observed_sequence_to_days(
            item["points"],
            base_centers,
            fitted[:, column],
            printed_days,
            row_pitch,
            (row_pitch * 0.05, row_pitch * 0.48),
        )
        if aligned is None:
            continue
        mapping, raw_offset = aligned
        if len(mapping) < 2:
            continue
        validated_tracks[column] = item
        dot_alignments[column] = mapping
        minimum_dot_coverage = max(10, round(printed_days * 0.34))
        if len(mapping) < minimum_dot_coverage:
            continue
        accepted_tracks[column] = item
        raw_dot_offsets.append(raw_offset)

    synthesis_tracks = (
        list(accepted_tracks.values())
        if len(accepted_tracks) >= 2
        else list(raw_tracks.values())
    )
    consensus_ratio = weighted_median(
        [float(item["ratio"]) for item in synthesis_tracks],
        [float(item["weight"]) for item in synthesis_tracks],
    )
    consensus_slope = weighted_median(
        [float(item["normalized_slope"]) for item in synthesis_tracks],
        [float(item["weight"]) for item in synthesis_tracks],
    )
    if raw_dot_offsets:
        dot_offset = float(np.median(raw_dot_offsets))
        dot_offset = min(row_pitch * 0.45, max(row_pitch * 0.08, dot_offset))
    else:
        dot_offset = row_pitch * 0.25

    for column in range(1, EXPECTED_COLUMN_COUNT):
        printed_days = maximum_printed_day(column)
        left = geometry.vertical_rules[column] + 5
        right = geometry.vertical_rules[column + 1] - 5
        width = max(1.0, right - left)
        if column in validated_tracks:
            decimal_model = validated_tracks[column]["model"]
        else:
            slope = consensus_slope * width / body_span
            midpoint_x = left + consensus_ratio * width
            decimal_model = (
                float(slope),
                float(midpoint_x - slope * body_midpoint),
            )

        dot_mapping = dot_alignments.get(column, {})
        tokens = detect_integer_token_centers_left_of_decimal(
            page_ink,
            geometry,
            column,
            decimal_model,
            row_pitch,
            min(body_bottom, round(statistics_samples[column]) - 2),
        )
        token_alignment = align_observed_sequence_to_days(
            tokens,
            base_centers,
            fitted[:, column],
            printed_days,
            row_pitch,
            (-row_pitch * 0.34, row_pitch * 0.34),
        )
        token_mapping = token_alignment[0] if token_alignment is not None else {}

        observed: dict[
            int, tuple[float, str, float, float, dict[str, Any]]
        ] = {}
        for day_index, point in dot_mapping.items():
            center_y = float(point["y"] - dot_offset)
            occupancy = inspect_decimal_slot_occupancy(
                page_ink,
                geometry,
                column,
                decimal_model,
                center_y,
                row_pitch,
            )
            observed[day_index] = (
                center_y,
                "decimal-dot",
                float(point["x"]),
                float(point["y"]),
                occupancy,
            )
        for day_index, token in token_mapping.items():
            if day_index in observed:
                # The two independent mechanisms must agree on the same row.
                # A disagreement rejects the weaker token, never the date.
                if abs(token["y"] - observed[day_index][0]) > row_pitch * 0.36:
                    continue
                continue
            decimal_x = decimal_model[0] * token["y"] + decimal_model[1]
            occupancy = inspect_decimal_slot_occupancy(
                page_ink,
                geometry,
                column,
                decimal_model,
                float(token["y"]),
                row_pitch,
            )
            if occupancy["layout"] == "integer-left-only":
                kind = "integer-left-of-decimal-track"
            elif occupancy["layout"] in (
                "decimal-slot-and-fraction",
                "faint-decimal-dispersed-fraction",
            ):
                kind = "decimal-layout-occupancy"
            elif (
                any(value < day_index for value in dot_mapping)
                and any(value > day_index for value in dot_mapping)
            ):
                kind = "integer-between-decimal-anchors"
            else:
                kind = "left-token-via-table-decimal-consensus"
            observed[day_index] = (
                float(token["y"]),
                kind,
                float(decimal_x),
                float(token["y"]),
                occupancy,
            )

        observed_days = sorted(observed)
        if len(observed_days) < 3:
            set_calendar_virtual_anchors(column, printed_days)
            continue
        observed_values = np.asarray(
            [observed[index][0] for index in observed_days], dtype=float
        )
        observed_displacement = (
            observed_values - base_centers[np.asarray(observed_days, dtype=int)]
        )
        interpolated_displacement = np.interp(
            np.arange(EXPECTED_DAY_COUNT, dtype=float),
            np.asarray(observed_days, dtype=float),
            observed_displacement,
        )
        smooth_displacement = cv2.GaussianBlur(
            interpolated_displacement[:, None].astype(np.float32),
            (1, 5),
            sigmaX=0.0,
            sigmaY=0.75,
        ).reshape(-1).astype(float)
        consensus_curve = base_centers + smooth_displacement
        first_observed = observed_days[0]
        last_observed = observed_days[-1]
        for day_index in range(printed_days):
            if day_index in observed:
                fitted[day_index, column] = (
                    observed[day_index][0] * 0.84
                    + consensus_curve[day_index] * 0.16
                )
            elif first_observed < day_index < last_observed:
                fitted[day_index, column] = (
                    consensus_curve[day_index] * 0.72
                    + fitted[day_index, column] * 0.28
                )

        # Calendar-invalid trailing rows are geometry-only control points.
        # They must never attach to ink.  Continue the exact day-column layout
        # gap from the last real date so the final valid cell still receives a
        # stable lower boundary (Feb 30/31 and day 31 in 30-day months).
        set_calendar_virtual_anchors(column, printed_days)

        # A valid daily center cannot cross the statistics boundary.  This is
        # a form-level invariant, not a station-specific exception.
        maximum_center = body_bottom - max(2.0, row_pitch * 0.12)
        fitted[:printed_days, column] = np.minimum(
            fitted[:printed_days, column], maximum_center
        )
        for day_index in observed_days:
            center_y, method, point_x, point_y, occupancy = observed[day_index]
            diagnostics[column].append(
                {
                    "month": column,
                    "day": day_index + 1,
                    "kind": method,
                    "point_x": float(point_x),
                    "point_y": float(point_y),
                    "row_center_y": float(center_y),
                    "decimal_x": float(
                        decimal_model[0] * point_y + decimal_model[1]
                    ),
                    "dot_to_center_offset": float(dot_offset),
                    "slot_layout": str(occupancy["layout"]),
                    "slot_ink": float(occupancy["slot_ink"]),
                    "left_ink": float(occupancy["left_ink"]),
                    "right_ink": float(occupancy["right_ink"]),
                    "ink_dispersion": float(occupancy["dispersion"]),
                }
            )

    geometry.month_decimal_anchors = diagnostics
    geometry.virtual_month_anchors = virtual_diagnostics
    if any(diagnostics[month] for month in range(1, EXPECTED_COLUMN_COUNT)):
        if "month-decimal" not in geometry.anchor_method:
            geometry.anchor_method += "+month-decimal-and-integer"
    return fitted


def fit_multicolumn_row_curves(
    image: np.ndarray, geometry: PageGeometry
) -> None:
    """Independently re-anchor every month column to its local ink rows.

    A column-wide correlation first estimates only that column's coarse shift.
    Three local refinement passes then attach each ordered day to a real ink
    band inside a non-overlapping row window.  No neighbouring-column result or
    statistics boundary can move the search window, so drift cannot accumulate.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    page_ink = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]
    centers_x = [
        (geometry.vertical_rules[index] + geometry.vertical_rules[index + 1])
        / 2
        for index in range(EXPECTED_COLUMN_COUNT)
    ]
    base_centers = np.asarray(
        [anchor.center for anchor in geometry.row_anchors], dtype=float
    )
    ordinary_gaps = np.diff(base_centers)
    ordinary_gaps = ordinary_gaps[
        ordinary_gaps <= np.percentile(ordinary_gaps, 70)
    ]
    row_pitch = float(np.median(ordinary_gaps)) if ordinary_gaps.size else 10.0
    row_pitch = max(4.0, row_pitch)
    fit_dynamic_vertical_rule_models(image, geometry, row_pitch)
    geometry.column_center_x = [
        (
            vertical_rule_x(geometry, column, float(np.median(base_centers)))
            + vertical_rule_x(geometry, column + 1, float(np.median(base_centers)))
        )
        / 2
        for column in range(EXPECTED_COLUMN_COUNT)
    ]
    boundary_radius = max(7, round(row_pitch * 0.75))
    geometry.table_top_curve = fit_horizontal_boundary_samples(
        gray,
        geometry,
        geometry.column_center_x,
        float(geometry.table_top),
        boundary_radius,
    )
    geometry.header_curve = fit_horizontal_boundary_samples(
        gray,
        geometry,
        geometry.column_center_x,
        float(geometry.header_bottom),
        boundary_radius,
    )
    # Daily ink must never borrow evidence from the first statistics row.
    # This is especially important for day 31, which has no following date to
    # close its search window.
    # This is only a generous global ceiling.  Each column is subsequently
    # bounded by its independently tracked statistics separator curve.
    body_bottom = min(
        geometry.height - 2,
        max(
            round(base_centers[-1] + row_pitch * 1.45),
            round(geometry.statistics_top + row_pitch * 1.20),
        ),
    )
    preliminary_lower_bounds = [
        float(base_centers[29 if column in (4, 6, 9, 11) else 30])
        + max(3.0, row_pitch * 0.42)
        for column in range(EXPECTED_COLUMN_COUNT)
    ]
    preliminary_lower_bounds[2] = (
        float(base_centers[28]) + max(3.0, row_pitch * 0.42)
    )
    statistics_samples = fit_statistics_separator_samples(
        gray,
        page_ink,
        geometry,
        geometry.column_center_x,
        row_pitch,
        preliminary_lower_bounds,
    )

    observations = np.full(
        (EXPECTED_DAY_COUNT, EXPECTED_COLUMN_COUNT), np.nan, dtype=float
    )
    weights = np.zeros_like(observations)
    observations[:, 0] = base_centers
    weights[:, 0] = 5.0

    def maximum_printed_day(column: int) -> int:
        if column == 2:
            return 29
        if column in (4, 6, 9, 11):
            return 30
        return 31

    for column in range(1, EXPECTED_COLUMN_COUNT):
        reference_y = float(np.median(base_centers))
        left = vertical_rule_x(geometry, column, reference_y) + 6
        right = vertical_rule_x(geometry, column + 1, reference_y) - 6
        column_width = max(1, round(right - left))
        column_body_top = max(
            geometry.header_bottom + 2,
            round(geometry.header_curve[column]) + 2
            if geometry.header_curve
            else geometry.header_bottom + 2,
        )
        column_body_bottom = min(body_bottom, round(statistics_samples[column]) - 2)
        projection = np.zeros(geometry.height, dtype=np.int32)
        for y in range(column_body_top, max(column_body_top, column_body_bottom)):
            dynamic_left = max(
                0, round(vertical_rule_x(geometry, column, y)) + 6
            )
            dynamic_right = min(
                geometry.width,
                round(vertical_rule_x(geometry, column + 1, y)) - 6,
            )
            if dynamic_right > dynamic_left:
                projection[y] = int(
                    np.count_nonzero(page_ink[y, dynamic_left:dynamic_right])
                )
        printed_days = maximum_printed_day(column)

        # Determine a robust column-local shift from the repeated 31-row
        # rhythm.  The window is deliberately narrower than half a row, so a
        # neighbouring date cannot improve the correlation score.
        half_score_window = max(2, round(row_pitch * 0.28))
        # Repeated daily rows make a full-row shift look deceptively similar
        # in correlation.  An independent month seed must stay inside the
        # identity interval of its day-column anchor; otherwise day 1 can lock
        # onto day 2 and shift the entire month.  Larger real bends are then
        # recovered gradually by the per-day deformation refinement below.
        shift_candidates = np.linspace(-row_pitch * 0.65, row_pitch * 0.65, 105)
        shift_scores: list[float] = []
        for shift in shift_candidates:
            score = 0.0
            supported = 0
            for center in base_centers[:printed_days] + shift:
                top = max(column_body_top, round(center) - half_score_window)
                bottom = min(
                    column_body_bottom,
                    round(center) + half_score_window + 1,
                )
                if bottom <= top:
                    continue
                peak = float(np.max(projection[top:bottom]))
                score += min(peak, column_width * 0.35)
                supported += peak >= 2
            identity_penalty = abs(shift) / row_pitch * column_width * 0.08
            shift_scores.append(
                score + supported * column_width * 0.015 - identity_penalty
            )
        shift = float(shift_candidates[int(np.argmax(shift_scores))])
        predicted = base_centers + shift
        final_centers = np.full(EXPECTED_DAY_COUNT, np.nan, dtype=float)
        final_weights = np.zeros(EXPECTED_DAY_COUNT, dtype=float)

        for _ in range(3):
            measured = np.full(EXPECTED_DAY_COUNT, np.nan, dtype=float)
            evidence_weights = np.zeros(EXPECTED_DAY_COUNT, dtype=float)
            for day_index in range(printed_days):
                if day_index == 0:
                    local_top = column_body_top
                else:
                    local_top = round(
                        (predicted[day_index - 1] + predicted[day_index]) / 2
                    )
                if day_index == EXPECTED_DAY_COUNT - 1:
                    local_bottom = column_body_bottom
                else:
                    local_bottom = round(
                        (predicted[day_index] + predicted[day_index + 1]) / 2
                    )
                local_top = max(0, local_top)
                local_bottom = min(column_body_bottom, local_bottom)
                if local_bottom <= local_top + 1:
                    continue
                values = projection[local_top:local_bottom].astype(float)
                peak = float(np.max(values)) if values.size else 0.0
                if peak < 2:
                    continue
                active = values >= max(1.0, peak * 0.12)
                bands = active_bands(active, local_top)
                choices: list[tuple[float, float, float]] = []
                for top, bottom in bands:
                    height = bottom - top + 1
                    band_values = projection[top : bottom + 1].astype(float)
                    band_peak = float(np.max(band_values))
                    total = float(np.sum(band_values))
                    if not (2 <= height <= row_pitch * 0.90):
                        continue
                    if band_peak >= column_width * 0.72 or total <= 0:
                        continue
                    y_values = np.arange(top, bottom + 1, dtype=float)
                    centroid = float(
                        np.average(y_values, weights=band_values)
                    )
                    evidence = min(
                        4.0,
                        0.5
                        + band_peak / max(1.0, column_width * 0.10)
                        + total / max(1.0, column_width * row_pitch * 0.18),
                    )
                    selection_cost = (
                        abs(centroid - predicted[day_index]) / row_pitch
                        - evidence * 0.035
                    )
                    choices.append((selection_cost, centroid, evidence))
                if choices:
                    _, centroid, evidence = min(choices, key=lambda item: item[0])
                    measured[day_index] = centroid
                    evidence_weights[day_index] = evidence

            valid = np.isfinite(measured)
            if np.count_nonzero(valid) < max(5, printed_days // 2):
                break
            displacement = measured[valid] - base_centers[valid]
            interpolated = np.interp(
                np.arange(EXPECTED_DAY_COUNT, dtype=float),
                np.flatnonzero(valid).astype(float),
                displacement,
            )
            smooth = cv2.GaussianBlur(
                interpolated[:, None].astype(np.float32),
                (1, 5),
                sigmaX=0.0,
                sigmaY=0.9,
            ).reshape(-1).astype(float)
            predicted = base_centers + smooth
            final_centers = measured
            final_weights = evidence_weights

        valid = np.isfinite(final_centers)
        observations[valid, column] = final_centers[valid]
        weights[valid, column] = final_weights[valid]

    displacement = observations - base_centers[:, None]
    filled_displacement = np.zeros_like(displacement)
    day_indices = np.arange(EXPECTED_DAY_COUNT, dtype=float)
    for column in range(EXPECTED_COLUMN_COUNT):
        valid = np.isfinite(displacement[:, column])
        if np.count_nonzero(valid) >= 2:
            filled_displacement[:, column] = np.interp(
                day_indices,
                day_indices[valid],
                displacement[valid, column],
            )
        elif np.count_nonzero(valid) == 1:
            filled_displacement[:, column] = float(displacement[valid, column][0])

    # A paper fold changes smoothly down the page.  A one-row displacement
    # jump is therefore a wrong glyph/row attachment, not real geometry.
    vertical_reference = np.zeros_like(filled_displacement)
    for column in range(EXPECTED_COLUMN_COUNT):
        for day_index in range(EXPECTED_DAY_COUNT):
            top = max(0, day_index - 2)
            bottom = min(EXPECTED_DAY_COUNT, day_index + 3)
            vertical_reference[day_index, column] = float(
                np.median(filled_displacement[top:bottom, column])
            )
    vertical_reference = cv2.GaussianBlur(
        vertical_reference.astype(np.float32),
        (1, 5),
        sigmaX=0.0,
        sigmaY=0.85,
    ).astype(float)
    valid = np.isfinite(displacement)
    outlier = valid & (
        np.abs(displacement - vertical_reference) > max(2.5, row_pitch * 0.24)
    )
    cleaned_displacement = vertical_reference.copy()
    trusted = valid & ~outlier
    cleaned_displacement[trusted] = (
        displacement[trusted] * 0.72 + vertical_reference[trusted] * 0.28
    )
    smooth_displacement = cv2.GaussianBlur(
        cleaned_displacement.astype(np.float32),
        (3, 3),
        sigmaX=0.50,
        sigmaY=0.65,
    ).astype(float)
    fitted_displacement = (
        cleaned_displacement * 0.82 + smooth_displacement * 0.18
    )
    fitted_displacement[:, 0] = 0.0
    fitted = base_centers[:, None] + fitted_displacement

    pre_anchor_fitted = fitted.copy()
    fitted = apply_month_decimal_anchors(
        image,
        page_ink,
        geometry,
        base_centers,
        fitted,
        row_pitch,
        body_bottom,
        statistics_samples,
    )

    # A photographed sheet bends smoothly in both directions.  Decimal/token
    # evidence is rejected locally when it creates a one-day or one-column
    # displacement jump; such a jump is a sequence attachment error, never a
    # physical fold.  The repair uses the 2-D neighbourhood and the independent
    # ink projection prior, so no station/page-specific thresholds are needed.
    displacement = fitted - base_centers[:, None]
    prior_displacement = pre_anchor_fitted - base_centers[:, None]
    for _ in range(3):
        neighbourhood = cv2.medianBlur(
            displacement.astype(np.float32), 3
        ).astype(float)
        residual = np.abs(displacement - neighbourhood)
        invalid = residual > max(3.0, row_pitch * 0.32)
        invalid[:, 0] = False
        displacement[invalid] = (
            neighbourhood[invalid] * 0.68
            + prior_displacement[invalid] * 0.32
        )
    vertical_cap = max(3.0, row_pitch * 0.36)
    horizontal_cap = max(3.0, row_pitch * 0.48)
    for column in range(1, EXPECTED_COLUMN_COUNT):
        for day_index in range(1, EXPECTED_DAY_COUNT):
            delta = displacement[day_index, column] - displacement[
                day_index - 1, column
            ]
            if abs(delta) > vertical_cap:
                candidate = neighbourhood[day_index, column]
                displacement[day_index, column] = (
                    candidate * 0.75
                    + prior_displacement[day_index, column] * 0.25
                )
    for day_index in range(EXPECTED_DAY_COUNT):
        for column in range(1, EXPECTED_COLUMN_COUNT):
            delta = displacement[day_index, column] - displacement[
                day_index, column - 1
            ]
            if abs(delta) > horizontal_cap:
                displacement[day_index, column] = (
                    neighbourhood[day_index, column] * 0.72
                    + prior_displacement[day_index, column] * 0.28
                )
    # Project onto a bounded-slope surface in both directions.  Averaging the
    # forward and backward projections avoids propagating whichever side of a
    # detected step happened to be visited first.
    horizontal_surface_cap = max(2.5, row_pitch * 0.34)
    for day_index in range(EXPECTED_DAY_COUNT):
        forward = displacement[day_index].copy()
        backward = displacement[day_index].copy()
        for column in range(1, EXPECTED_COLUMN_COUNT):
            forward[column] = np.clip(
                forward[column],
                forward[column - 1] - horizontal_surface_cap,
                forward[column - 1] + horizontal_surface_cap,
            )
        for column in range(EXPECTED_COLUMN_COUNT - 2, -1, -1):
            backward[column] = np.clip(
                backward[column],
                backward[column + 1] - horizontal_surface_cap,
                backward[column + 1] + horizontal_surface_cap,
            )
        displacement[day_index] = (forward + backward) / 2
    vertical_surface_cap = max(2.5, row_pitch * 0.34)
    for column in range(1, EXPECTED_COLUMN_COUNT):
        forward = displacement[:, column].copy()
        backward = displacement[:, column].copy()
        for day_index in range(1, EXPECTED_DAY_COUNT):
            forward[day_index] = np.clip(
                forward[day_index],
                forward[day_index - 1] - vertical_surface_cap,
                forward[day_index - 1] + vertical_surface_cap,
            )
        for day_index in range(EXPECTED_DAY_COUNT - 2, -1, -1):
            backward[day_index] = np.clip(
                backward[day_index],
                backward[day_index + 1] - vertical_surface_cap,
                backward[day_index + 1] + vertical_surface_cap,
            )
        displacement[:, column] = (forward + backward) / 2
    displacement[:, 0] = 0.0
    for day_index in range(EXPECTED_DAY_COUNT):
        for column in range(1, EXPECTED_COLUMN_COUNT):
            displacement[day_index, column] = np.clip(
                displacement[day_index, column],
                displacement[day_index, column - 1] - horizontal_surface_cap,
                displacement[day_index, column - 1] + horizontal_surface_cap,
            )
    fitted = base_centers[:, None] + displacement

    minimum_gap = max(2.0, row_pitch * 0.20)
    for column in range(EXPECTED_COLUMN_COUNT):
        for day_index in range(1, EXPECTED_DAY_COUNT):
            fitted[day_index, column] = max(
                fitted[day_index, column],
                fitted[day_index - 1, column] + minimum_gap,
            )
        for day_index in range(EXPECTED_DAY_COUNT - 2, -1, -1):
            fitted[day_index, column] = min(
                fitted[day_index, column],
                fitted[day_index + 1, column] - minimum_gap,
            )

    # A separator estimate above the final valid glyph band is necessarily a
    # digit stroke, not the statistics divider.  Enforce the independently
    # observed last-row lower edge before it is used as a crop boundary.  The
    # physical divider is also continuous across adjacent months, so isolated
    # Hough minima are replaced by a lower-bounded, bounded-slope curve.
    separator_lower_bounds: list[float] = []
    for column in range(EXPECTED_COLUMN_COUNT):
        printed_days = 31 if column == 0 else maximum_printed_day(column)
        last_center = fitted[printed_days - 1, column]
        ink_lower_bound = last_center + max(3.0, row_pitch * 0.42)
        separator_lower_bounds.append(float(ink_lower_bound))
    statistics_samples = fit_statistics_separator_samples(
        gray,
        page_ink,
        geometry,
        centers_x,
        row_pitch,
        separator_lower_bounds,
    )
    for column in range(EXPECTED_COLUMN_COUNT):
        statistics_samples[column] = min(
            geometry.height - 2.0,
            max(float(statistics_samples[column]), separator_lower_bounds[column]),
        )
    separator = np.asarray(statistics_samples, dtype=float)
    lower_bounds = np.asarray(separator_lower_bounds, dtype=float)
    local_median = cv2.medianBlur(
        separator.astype(np.float32)[None, :], 3
    ).reshape(-1).astype(float)
    suspicious = np.abs(separator - local_median) > row_pitch * 0.42
    separator[suspicious] = np.maximum(
        lower_bounds[suspicious], local_median[suspicious]
    )
    separator_cap = max(3.0, row_pitch * 0.42)
    for _ in range(3):
        forward = separator.copy()
        backward = separator.copy()
        for column in range(1, EXPECTED_COLUMN_COUNT):
            forward[column] = max(
                lower_bounds[column],
                float(
                    np.clip(
                        forward[column],
                        forward[column - 1] - separator_cap,
                        forward[column - 1] + separator_cap,
                    )
                ),
            )
        for column in range(EXPECTED_COLUMN_COUNT - 2, -1, -1):
            backward[column] = max(
                lower_bounds[column],
                float(
                    np.clip(
                        backward[column],
                        backward[column + 1] - separator_cap,
                        backward[column + 1] + separator_cap,
                    )
                ),
            )
        separator = np.maximum(lower_bounds, (forward + backward) / 2)
    statistics_samples = separator.astype(float).tolist()

    center_reference_y = float(np.median(base_centers))
    geometry.column_center_x = [
        (
            vertical_rule_x(geometry, column, center_reference_y)
            + vertical_rule_x(geometry, column + 1, center_reference_y)
        )
        / 2
        for column in range(EXPECTED_COLUMN_COUNT)
    ]
    geometry.row_curve_centers = fitted.tolist()
    geometry.statistics_curve = statistics_samples


def curve_values_at_rules(
    geometry: PageGeometry, values: list[float]
) -> list[float]:
    if not geometry.column_center_x or not values:
        return [float(values[0] if values else geometry.statistics_top)] * len(
            geometry.vertical_rules
        )
    source_x = np.asarray(geometry.column_center_x, dtype=float)
    source_y = np.asarray(values, dtype=float)
    query_x = np.asarray(geometry.vertical_rules, dtype=float)
    if len(source_x) >= 3 and np.all(np.diff(source_x) > 0):
        # Shape-preserving cubic interpolation follows photographic bending
        # without the overshoot produced by an unconstrained cubic spline.
        interpolated = PchipInterpolator(
            source_x, source_y, extrapolate=True
        )(query_x).astype(float)
        interpolated[query_x <= source_x[0]] = source_y[0]
        interpolated[query_x >= source_x[-1]] = source_y[-1]
        return interpolated.tolist()
    return np.interp(query_x, source_x, source_y).tolist()


def curved_cell_polygon(
    geometry: PageGeometry, month: int, day_index: int
) -> list[list[float]]:
    row_centers = geometry.row_curve_centers
    if not row_centers:
        top, bottom = anchor_row_bounds(
            geometry.row_anchors,
            day_index,
            geometry.header_bottom + 2,
            geometry.statistics_top - 1,
        )
        return [
            [geometry.vertical_rules[month], top],
            [geometry.vertical_rules[month + 1], top],
            [geometry.vertical_rules[month + 1], bottom],
            [geometry.vertical_rules[month], bottom],
        ]
    if day_index == 0:
        top_values = (
            [float(value + 2.0) for value in geometry.header_curve]
            if len(geometry.header_curve) == EXPECTED_COLUMN_COUNT
            else [float(geometry.header_bottom + 2)] * EXPECTED_COLUMN_COUNT
        )
    else:
        top_values = [
            (row_centers[day_index - 1][column] + row_centers[day_index][column])
            / 2
            for column in range(EXPECTED_COLUMN_COUNT)
        ]
    if day_index == EXPECTED_DAY_COUNT - 1:
        bottom_values = [
            row_centers[-1][column]
            + max(
                4.0,
                (row_centers[-1][column] - row_centers[-2][column]) * 0.52,
            )
            for column in range(EXPECTED_COLUMN_COUNT)
        ]
    else:
        bottom_values = [
            (row_centers[day_index][column] + row_centers[day_index + 1][column])
            / 2
            for column in range(EXPECTED_COLUMN_COUNT)
        ]
    top_rules = curve_values_at_rules(geometry, top_values)
    bottom_rules = curve_values_at_rules(geometry, bottom_values)
    month_lengths = (
        geometry.month_lengths
        if len(geometry.month_lengths) == 12
        else LEAP_MONTH_LENGTHS
    )
    if (
        day_index == month_lengths[month - 1] - 1
        and geometry.statistics_curve
    ):
        statistics_rules = curve_values_at_rules(
            geometry, geometry.statistics_curve
        )
        # The separator itself is the only non-ink lower boundary for the
        # final valid date.  Calendar-virtual centers are diagnostic controls
        # and never determine this crop edge.
        bottom_rules[month] = statistics_rules[month] - 1.0
        bottom_rules[month + 1] = statistics_rules[month + 1] - 1.0
    top_left_x = vertical_rule_x(geometry, month, top_rules[month])
    top_right_x = vertical_rule_x(geometry, month + 1, top_rules[month + 1])
    bottom_right_x = vertical_rule_x(
        geometry, month + 1, bottom_rules[month + 1]
    )
    bottom_left_x = vertical_rule_x(geometry, month, bottom_rules[month])
    return [
        [top_left_x, top_rules[month]],
        [top_right_x, top_rules[month + 1]],
        [bottom_right_x, bottom_rules[month + 1]],
        [bottom_left_x, bottom_rules[month]],
    ]


def prepare_curved_cell(
    image: np.ndarray, polygon: list[list[float]], horizontal_inset: int
) -> np.ndarray | None:
    top_left = np.asarray(polygon[0], dtype=np.float32)
    top_right = np.asarray(polygon[1], dtype=np.float32)
    bottom_right = np.asarray(polygon[2], dtype=np.float32)
    bottom_left = np.asarray(polygon[3], dtype=np.float32)
    top_left[0] += horizontal_inset
    bottom_left[0] += horizontal_inset
    top_right[0] -= horizontal_inset
    bottom_right[0] -= horizontal_inset
    top_width = float(np.linalg.norm(top_right - top_left))
    bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
    if min(top_width, bottom_width) <= 3:
        return None
    output_width = max(4, round((top_width + bottom_width) / 2))
    output_height = max(
        4,
        round(
            (
                float(np.linalg.norm(bottom_left - top_left))
                + float(np.linalg.norm(bottom_right - top_right))
            )
            / 2
        ),
    )
    fractions_x = np.linspace(0.0, 1.0, output_width, dtype=np.float32)[
        None, :, None
    ]
    fractions_y = np.linspace(0.0, 1.0, output_height, dtype=np.float32)[:, None]
    top_edge = top_left[None, None, :] * (1.0 - fractions_x) + top_right[
        None, None, :
    ] * fractions_x
    bottom_edge = bottom_left[None, None, :] * (
        1.0 - fractions_x
    ) + bottom_right[None, None, :] * fractions_x
    fractions_y_3d = fractions_y[:, :, None]
    mapped = top_edge * (1.0 - fractions_y_3d) + bottom_edge * fractions_y_3d
    map_x = mapped[:, :, 0].astype(np.float32)
    map_y = mapped[:, :, 1].astype(np.float32)
    cell = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return prepare_cell(cell, (0, 0, output_width, output_height))


def normalized_numeric_text(text: str) -> str:
    normalized = (
        str(text)
        .strip()
        .translate(NUMERIC_TRANSLATION)
        .replace("，", ".")
        .replace(",", ".")
        .replace("：", ".")
        .replace(":", ".")
    )
    return re.sub(r"\s+", "", normalized)


def excel_literal_text(value: Any) -> str:
    """Keep OCR diagnostics as text even when they resemble Excel formulas/errors."""

    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@", "#")):
        return "'" + text
    return text


def parse_flow(text: str) -> float | None:
    cleaned = normalized_numeric_text(text)
    matches = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not matches:
        return None
    candidate = max(matches, key=lambda item: (len(item), "." in item))
    try:
        return float(candidate)
    except ValueError:
        return None


def cell_has_ink(cell: np.ndarray | None) -> bool:
    if cell is None or cell.size == 0:
        return False
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    return int(np.count_nonzero(ink)) >= max(8, round(ink.size * 0.008))


def infer_month_lengths(
    image: np.ndarray, geometry: PageGeometry
) -> list[int]:
    day_29_top, day_29_bottom = anchor_row_bounds(
        geometry.row_anchors,
        28,
        geometry.header_bottom + 2,
        geometry.statistics_top - 1,
    )
    february_cell = prepare_cell(
        image,
        (
            geometry.vertical_rules[2] + 2,
            day_29_top,
            geometry.vertical_rules[3] - 2,
            day_29_bottom,
        ),
    )
    return LEAP_MONTH_LENGTHS.copy() if cell_has_ink(february_cell) else COMMON_MONTH_LENGTHS.copy()


def segment_title_crops(
    image: np.ndarray, geometry: PageGeometry
) -> tuple[np.ndarray, list[np.ndarray]]:
    height = geometry.height
    span = geometry.table_right - geometry.table_left
    top = max(0, geometry.table_top - round(height * 0.060))
    bottom = max(top + 10, geometry.table_top - round(height * 0.010))
    left = max(0, geometry.table_left + round(span * 0.08))
    right = min(geometry.width, geometry.table_right - round(span * 0.08))
    crop = image[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    ink = cv2.morphologyEx(
        ink,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    closed = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(7, round(crop.shape[1] * 0.010)), 3)
        ),
    )
    projection = np.count_nonzero(closed, axis=0)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate((projection >= 2).tolist()):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= 5:
                runs.append((start, index - 1))
            start = None
    if start is not None and len(projection) - start >= 5:
        runs.append((start, len(projection) - 1))

    merged: list[list[int]] = []
    merge_gap = max(9, round(crop.shape[1] * 0.014))
    for run_left, run_right in runs:
        if not merged or run_left - merged[-1][1] > merge_gap:
            merged.append([run_left, run_right])
        else:
            merged[-1][1] = run_right
    parts: list[np.ndarray] = []
    for run_left, run_right in merged:
        part = crop[:, max(0, run_left - 4) : min(crop.shape[1], run_right + 5)]
        if part.shape[1] >= 8:
            parts.append(part)
    return crop, parts


def recognize_page_heading(
    image: np.ndarray,
    geometry: PageGeometry,
    recognizer: TextRecognition,
) -> dict[str, Any]:
    title_crop, parts = segment_title_crops(image, geometry)
    crops = [title_crop] + parts
    predictions = list(recognizer.predict(input=crops, batch_size=len(crops)))
    recognized = [result_value(prediction) for prediction in predictions]
    full_text, full_confidence = recognized[0]
    part_results = [
        {"text": text, "confidence": confidence}
        for text, confidence in recognized[1:]
        if str(text).strip()
    ]

    station = ""
    river = ""
    table_number = ""
    for index, item in enumerate(part_results):
        text = re.sub(r"\s+", "", str(item["text"]))
        if not table_number and re.fullmatch(r"\d{1,5}", text):
            table_number = text
        if text.endswith("站") and "流量" not in text:
            station = text
            if index > 0:
                previous = re.sub(
                    r"\s+", "", str(part_results[index - 1]["text"])
                )
                if re.fullmatch(r"[\u4e00-\u9fff·（）()]+", previous):
                    river = previous

    compact = re.sub(r"\s+", "", full_text)
    if not table_number:
        match = re.match(r"(\d{1,5})", compact)
        table_number = match.group(1) if match else ""
    if not station:
        prefix = re.sub(r"^\d+", "", compact)
        prefix = re.split(r"逐日|平均流量", prefix)[0]
        match = re.search(r"([\u4e00-\u9fff·（）()]+站)$", prefix)
        station = match.group(1) if match else ""
    station = re.sub(r"\s+", "", station).replace("(", "（").replace(")", "）")
    if not station:
        station = f"第{geometry.page}页站点"

    span = geometry.table_right - geometry.table_left
    metadata_crop = image[
        max(0, geometry.table_top - round(geometry.height * 0.035)) : geometry.table_top,
        geometry.table_left + round(span * 0.62) : geometry.table_right,
    ]
    metadata_prediction = next(
        iter(recognizer.predict(input=[metadata_crop], batch_size=1))
    )
    metadata_text, metadata_confidence = result_value(metadata_prediction)
    area_match = re.search(
        r"集水面积\D*(\d+(?:\.\d+)?)", metadata_text.replace(" ", "")
    )
    catchment_area = float(area_match.group(1)) if area_match else None
    return {
        "table_number": table_number,
        "river": river,
        "station": station,
        "title_text": full_text,
        "title_confidence": full_confidence,
        "title_parts": part_results,
        "metadata_text": metadata_text,
        "metadata_confidence": metadata_confidence,
        "catchment_area_km2": catchment_area,
        "unit": "m³/s",
    }


def recognize_region_heading(
    source_image: np.ndarray,
    region: TableRegion,
    recognizer: TextRecognition,
) -> dict[str, Any]:
    """Read the title belonging to one of several tables on a photo page."""

    page_height, page_width = source_image.shape[:2]
    left, top, right, _ = region.source_bbox
    span = right - left
    title_crop = source_image[
        max(0, top - round(page_height * 0.038)) : max(1, top - round(page_height * 0.008)),
        max(0, left - round(span * 0.01)) : min(page_width, left + round(span * 0.76)),
    ]
    metadata_crops = [
        source_image[
            max(0, top - round(page_height * 0.025)) : max(
                1, top - round(page_height * 0.003)
            ),
            max(0, left + round(span * 0.62)) : min(page_width, right),
        ],
        source_image[
            max(0, top - round(page_height * 0.033)) : top,
            max(0, left + round(span * 0.72)) : min(page_width, right),
        ],
        source_image[
            max(0, top - round(page_height * 0.030)) : max(
                1, top - round(page_height * 0.005)
            ),
            max(0, left + round(span * 0.72)) : min(page_width, right),
        ],
    ]
    predictions = list(
        recognizer.predict(
            input=[title_crop, *metadata_crops], batch_size=4
        )
    )
    title_text, title_confidence = result_value(predictions[0])
    metadata_results = [result_value(item) for item in predictions[1:]]
    metadata_text = " | ".join(text for text, _ in metadata_results)
    metadata_confidence = max(confidence for _, confidence in metadata_results)
    compact = re.sub(r"\s+", "", str(title_text))
    table_match = re.match(r"(\d{1,5})", compact)
    table_number = table_match.group(1) if table_match else ""
    prefix = re.split(r"逐日|平均流量", re.sub(r"^\d+", "", compact))[0]
    river = ""
    station = ""
    river_match = re.match(
        r"(.+?(?:江|河|溪|沟|湖|渠|水库))(.+?站)$", prefix
    )
    if river_match:
        river, station = river_match.groups()
    else:
        station_match = re.search(r"([\u4e00-\u9fff·（）()]+站)$", prefix)
        station = station_match.group(1) if station_match else ""
    station = station.replace("(", "（").replace(")", "）")
    if not station:
        station = f"第{region.page}页表{region.table_index}站点"
    metadata_compact = re.sub(r"\s+", "", str(metadata_text))
    area_candidates = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?=k?m|k㎡|平方)",
        metadata_compact,
        re.I,
    )
    if not area_candidates:
        area_candidates = re.findall(
            r"集水面积\D*(\d+(?:\.\d+)?)", metadata_compact
        )
    area_text = max(area_candidates, key=len) if area_candidates else ""
    catchment_area = float(area_text) if area_text else None
    return {
        "table_number": table_number,
        "river": river,
        "station": station,
        "title_text": title_text,
        "title_confidence": title_confidence,
        "title_parts": [],
        "metadata_text": metadata_text,
        "metadata_confidence": metadata_confidence,
        "catchment_area_km2": catchment_area,
        "unit": "m³/s",
    }


def geometry_to_json(geometry: PageGeometry) -> dict[str, Any]:
    def anchor_payload(anchor: AnchorCandidate) -> dict[str, Any]:
        return {
            "top": anchor.top,
            "bottom": anchor.bottom,
            "bbox_center_y": round(anchor.center, 3),
            "ink_centroid_y": round(anchor.ink_centroid, 3),
            "support_columns": anchor.support_columns,
            "day_column_support": anchor.day_column_support,
            "score": round(anchor.score, 3),
        }

    return {
        "page": geometry.page,
        "table_index": geometry.table_index,
        "image_width": geometry.width,
        "image_height": geometry.height,
        "table_bbox": [
            geometry.table_left,
            geometry.table_top,
            geometry.table_right,
            geometry.statistics_top,
        ],
        "header_bottom": geometry.header_bottom,
        "statistics_top": geometry.statistics_top,
        "table_top_curve_y": [
            round(value, 3) for value in geometry.table_top_curve
        ],
        "header_curve_y": [round(value, 3) for value in geometry.header_curve],
        "vertical_rules": geometry.vertical_rules,
        "vertical_rule_models_x_by_y": [
            [round(float(value), 9) for value in coefficients]
            for coefficients in geometry.vertical_rule_models
        ],
        "horizontal_rules": geometry.horizontal_rules,
        "anchor_method": geometry.anchor_method,
        "candidate_anchor_count": len(geometry.candidate_anchors),
        "candidate_anchors": [
            anchor_payload(item) for item in geometry.candidate_anchors
        ],
        "selected_anchor_count": len(geometry.row_anchors),
        "selected_anchors": [
            {"day": index + 1, **anchor_payload(item)}
            for index, item in enumerate(geometry.row_anchors)
        ],
        "curve_model": "month-decimal-track-with-left-integer-fallback",
        "column_center_x": [
            round(value, 3) for value in geometry.column_center_x
        ],
        "statistics_curve_y": [
            round(value, 3) for value in geometry.statistics_curve
        ],
        "row_curve_centers": [
            {
                "day": day + 1,
                "y_by_column": [round(value, 3) for value in values],
            }
            for day, values in enumerate(geometry.row_curve_centers)
        ],
        "month_decimal_anchors": [
            {
                "month": month,
                "anchors": [
                    {
                        **item,
                        "point_x": round(float(item["point_x"]), 3),
                        "point_y": round(float(item["point_y"]), 3),
                        "row_center_y": round(
                            float(item["row_center_y"]), 3
                        ),
                        "decimal_x": round(float(item["decimal_x"]), 3),
                        "dot_to_center_offset": round(
                            float(item["dot_to_center_offset"]), 3
                        ),
                    }
                    for item in anchors
                ],
            }
            for month, anchors in enumerate(geometry.month_decimal_anchors)
            if month > 0 and anchors
        ],
        "virtual_calendar_anchors": [
            {
                **item,
                "row_center_y": round(float(item["row_center_y"]), 3),
            }
            for item in geometry.virtual_month_anchors
        ],
    }


def apply_corrected_labels(
    samples: list[dict[str, Any]], corrected_path: Path
) -> int:
    workbook = load_workbook(corrected_path, data_only=False)
    if "标签校正" not in workbook.sheetnames:
        raise ValueError("标签校正工作簿缺少“标签校正”工作表。")
    sheet = workbook["标签校正"]
    headers = {
        str(cell.value).strip(): index + 1
        for index, cell in enumerate(sheet[1])
        if cell.value is not None
    }
    required = {"样本ID", "正确标签", "纳入训练"}
    missing = required - set(headers)
    if missing:
        raise ValueError("标签校正工作簿缺少列：" + "、".join(sorted(missing)))
    corrections: dict[str, str] = {}
    for row in range(2, sheet.max_row + 1):
        sample_id = str(sheet.cell(row, headers["样本ID"]).value or "").strip()
        correct = str(sheet.cell(row, headers["正确标签"]).value or "").strip()
        include = str(sheet.cell(row, headers["纳入训练"]).value or "").strip()
        if sample_id and correct and include == "是":
            corrections[sample_id] = correct
    workbook.close()
    applied = 0
    for sample in samples:
        correct = corrections.get(sample["sample_id"])
        if not correct:
            continue
        value = parse_flow(correct)
        if value is None:
            raise ValueError(
                f"样本{sample['sample_id']}的正确标签不是有效流量：{correct}"
            )
        sample["correct_label"] = correct
        sample["include"] = "是"
        sample["selected_text"] = correct
        sample["value"] = value
        sample["value_source"] = "corrected-label"
        sample["note"] = "已采用人工校正标签"
        applied += 1
    return applied


def recognize_daily_cells(
    image: np.ndarray,
    geometry: PageGeometry,
    heading: dict[str, Any],
    primary_recognizer: TextRecognition,
    secondary_recognizer: TextRecognition | None,
    training_images_dir: Path,
    batch_size: int,
    low_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    month_lengths = infer_month_lengths(image, geometry)
    geometry.month_lengths = list(month_lengths)
    fit_multicolumn_row_curves(image, geometry)
    training_images_dir.mkdir(parents=True, exist_ok=True)
    crops: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []

    for day_index, day in enumerate(range(1, EXPECTED_DAY_COUNT + 1)):
        anchor = geometry.row_anchors[day_index]
        for month in range(1, 13):
            if day > month_lengths[month - 1]:
                continue
            column_width = (
                geometry.vertical_rules[month + 1]
                - geometry.vertical_rules[month]
            )
            cell_inset = max(3, round(column_width * 0.060))
            polygon = curved_cell_polygon(geometry, month, day_index)
            prepared = prepare_curved_cell(image, polygon, cell_inset)
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            bounds = [
                round(min(xs)) + cell_inset,
                round(min(ys)),
                round(max(xs)) - cell_inset,
                round(max(ys)),
            ]
            sample_id = (
                f"p{geometry.page:03d}_t{geometry.table_index:02d}_"
                f"m{month:02d}_d{day:02d}_flow"
            )
            image_name = f"{sample_id}.png"
            image_path = training_images_dir / image_name
            if prepared is None:
                prepared = np.full((80, 80, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(image_path), prepared)
            crops.append(prepared)
            samples.append(
                {
                    "sample_id": sample_id,
                    "page": geometry.page,
                    "panel": geometry.table_index,
                    "row": day,
                    "month": month,
                    "day": day,
                    "column": "流量(m³/s)",
                    "bounds": bounds,
                    "polygon": [
                        [round(x, 3), round(y, 3)] for x, y in polygon
                    ],
                    "anchor_y": round(anchor.center, 3),
                    "anchor_ink_centroid_y": round(anchor.ink_centroid, 3),
                    "anchor_support_columns": anchor.support_columns,
                    "image": f"images/{image_name}",
                    "station": heading["station"],
                }
            )

    primary_predictions = list(
        primary_recognizer.predict(input=crops, batch_size=batch_size)
    )
    secondary_predictions: list[Any] = []
    if secondary_recognizer is not None:
        secondary_predictions = list(
            secondary_recognizer.predict(input=crops, batch_size=batch_size)
        )
    if len(primary_predictions) != len(samples):
        raise RuntimeError("主OCR模型返回的流量候选数量与单元格数量不一致。")
    if secondary_recognizer is not None and len(secondary_predictions) != len(samples):
        raise RuntimeError("辅助OCR模型返回的流量候选数量与单元格数量不一致。")

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        primary_text, primary_confidence = result_value(primary_predictions[index])
        primary_value = parse_flow(primary_text)
        secondary_text = ""
        secondary_confidence = 0.0
        secondary_value: float | None = None
        if secondary_recognizer is not None:
            secondary_text, secondary_confidence = result_value(
                secondary_predictions[index]
            )
            secondary_value = parse_flow(secondary_text)

        conflict = (
            primary_value is not None
            and secondary_value is not None
            and abs(primary_value - secondary_value) > 1.0e-9
        )
        primary_has_leading_artifact = bool(
            re.match(r"^[|!Il]", str(primary_text).strip())
        )
        primary_digit_count = len(re.findall(r"\d", str(primary_text)))
        secondary_digit_count = len(re.findall(r"\d", str(secondary_text)))
        secondary_has_clearer_digits = (
            conflict
            and secondary_digit_count >= primary_digit_count + 1
            and secondary_confidence >= primary_confidence - 0.20
        )
        if primary_value is not None and (
            secondary_value is None
            or not conflict
            or (
                primary_confidence >= secondary_confidence
                and not primary_has_leading_artifact
                and not secondary_has_clearer_digits
            )
        ):
            selected_value = primary_value
            selected_text = primary_text
            value_source = "primary"
        elif secondary_value is not None:
            selected_value = secondary_value
            selected_text = secondary_text
            value_source = "secondary"
        else:
            selected_value = None
            selected_text = ""
            value_source = "unreadable"

        confidences = [primary_confidence]
        if secondary_recognizer is not None:
            confidences.append(secondary_confidence)
        average_confidence = float(np.mean(confidences))
        notes: list[str] = []
        include = "是"
        if selected_value is None:
            notes.append("两个模型均没有形成有效数值")
            include = "待确认"
        if conflict:
            notes.append("主模型与辅助模型数值冲突")
            include = "待确认"
        if average_confidence < low_confidence:
            notes.append("平均置信度低于阈值")
            include = "待确认"

        sample.update(
            {
                "ocr_text": primary_text,
                "primary_text": primary_text,
                "primary_value": primary_value,
                "primary_confidence": primary_confidence,
                "secondary_text": secondary_text,
                "secondary_value": secondary_value,
                "secondary_confidence": secondary_confidence,
                "selected_text": selected_text,
                "normalized_label": (
                    "" if selected_value is None else f"{selected_value:g}"
                ),
                "confidence": average_confidence,
                "correct_label": (
                    "" if selected_value is None else f"{selected_value:g}"
                ),
                "include": include,
                "note": "；".join(notes),
                "value": selected_value,
                "value_source": value_source,
                "model_conflict": conflict,
            }
        )
        rows.append(
            {
                "页": geometry.page,
                "表区": geometry.table_index,
                "月": sample["month"],
                "日": sample["day"],
                "流量(m³/s)": selected_value,
                "平均置信度": average_confidence,
                "主模型文本": primary_text,
                "主模型置信度": primary_confidence,
                "辅助模型文本": secondary_text,
                "辅助模型置信度": secondary_confidence,
                "值来源": value_source,
                "模型冲突": conflict,
                "待确认": include == "待确认",
                "备注": "；".join(notes),
                "样本ID": sample["sample_id"],
                "行锚点Y": sample["anchor_y"],
                "锚点墨迹重心Y": sample["anchor_ink_centroid_y"],
                "锚点支持列数": sample["anchor_support_columns"],
                "单元格边界": sample["bounds"],
                "单元格多边形": sample["polygon"],
                "图片": sample["image"],
            }
        )
    return rows, samples, month_lengths


def synchronize_rows_from_samples(
    rows: list[dict[str, Any]], samples: list[dict[str, Any]]
) -> None:
    by_id = {sample["sample_id"]: sample for sample in samples}
    for row in rows:
        sample = by_id[row["样本ID"]]
        row["流量(m³/s)"] = sample["value"]
        row["值来源"] = sample["value_source"]
        row["待确认"] = sample["include"] == "待确认"
        row["备注"] = sample["note"]


def draw_anchor_preview(
    image: np.ndarray, geometry: PageGeometry, output_path: Path
) -> None:
    preview = image.copy()
    top_rules = boundary_values_at_rules(
        geometry, geometry.table_top_curve, geometry.table_top
    )
    bottom_rules = boundary_values_at_rules(
        geometry, geometry.statistics_curve, geometry.statistics_top
    )
    top_points = np.asarray(
        [
            [round(vertical_rule_x(geometry, index, y)), round(y)]
            for index, y in enumerate(top_rules)
        ],
        dtype=np.int32,
    )
    bottom_points = np.asarray(
        [
            [round(vertical_rule_x(geometry, index, y)), round(y)]
            for index, y in enumerate(bottom_rules)
        ],
        dtype=np.int32,
    )
    cv2.polylines(preview, [top_points], False, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.polylines(preview, [bottom_points], False, (255, 0, 255), 2, cv2.LINE_AA)
    for rule_index in (0, len(geometry.vertical_rules) - 1):
        side_y = np.linspace(top_rules[rule_index], bottom_rules[rule_index], 80)
        side_points = np.asarray(
            [
                [round(vertical_rule_x(geometry, rule_index, y)), round(y)]
                for y in side_y
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            preview, [side_points], False, (255, 0, 255), 2, cv2.LINE_AA
        )
    for rule_index in range(len(geometry.vertical_rules)):
        sample_y = np.linspace(
            geometry.table_top, geometry.statistics_top, 80
        )
        points = np.asarray(
            [
                [round(vertical_rule_x(geometry, rule_index, y)), round(y)]
                for y in sample_y
            ],
            dtype=np.int32,
        )
        cv2.polylines(preview, [points], False, (255, 200, 0), 1, cv2.LINE_AA)
    header_rules = boundary_values_at_rules(
        geometry, geometry.header_curve, geometry.header_bottom
    )
    header_points = np.asarray(
        [
            [round(vertical_rule_x(geometry, index, y)), round(y)]
            for index, y in enumerate(header_rules)
        ],
        dtype=np.int32,
    )
    cv2.polylines(preview, [header_points], False, (0, 0, 255), 2, cv2.LINE_AA)
    if geometry.statistics_curve:
        statistics_points = np.asarray(
            [
                [round(x), round(y)]
                for x, y in zip(
                    geometry.column_center_x, geometry.statistics_curve
                )
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            preview, [statistics_points], False, (0, 0, 255), 2, cv2.LINE_AA
        )
    else:
        cv2.line(
            preview,
            (geometry.table_left, geometry.statistics_top),
            (geometry.table_right, geometry.statistics_top),
            (0, 0, 255),
            2,
        )
    selected_centers = {
        round(item.center): index + 1
        for index, item in enumerate(geometry.row_anchors)
    }
    for candidate in geometry.candidate_anchors:
        y = round(candidate.center)
        cv2.circle(preview, (geometry.table_left - 12, y), 4, (0, 215, 255), -1)
    for day_index, anchor in enumerate(geometry.row_anchors):
        y = round(anchor.center)
        day = selected_centers[y]
        if geometry.row_curve_centers:
            points = np.asarray(
                [
                    [
                        round(
                            (
                                vertical_rule_x(geometry, column, curve_y)
                                + vertical_rule_x(
                                    geometry, column + 1, curve_y
                                )
                            )
                            / 2
                        ),
                        round(curve_y),
                    ]
                    for column, curve_y in enumerate(
                        geometry.row_curve_centers[day_index]
                    )
                ],
                dtype=np.int32,
            )
            cv2.polylines(
                preview, [points], False, (0, 180, 0), 1, cv2.LINE_AA
            )
            for point_x, point_y in points:
                cv2.circle(
                    preview,
                    (int(point_x), int(point_y)),
                    2,
                    (255, 80, 0),
                    -1,
                )
        else:
            cv2.line(
                preview,
                (geometry.table_left, y),
                (geometry.table_right, y),
                (0, 180, 0),
                1,
            )
        cv2.circle(preview, (geometry.table_left - 12, y), 5, (0, 160, 0), -1)
        cv2.putText(
            preview,
            str(day),
            (max(2, geometry.table_left - 55), y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 100, 0),
            1,
            cv2.LINE_AA,
        )
    for month_anchors in geometry.month_decimal_anchors:
        for item in month_anchors:
            point = (
                round(float(item["point_x"])),
                round(float(item["point_y"])),
            )
            if item["kind"] == "decimal-dot":
                # Magenta rings are directly observed decimal points.
                cv2.circle(preview, point, 4, (255, 0, 255), 1, cv2.LINE_AA)
            else:
                # Yellow crosses are integer tokens found to the left of the
                # fitted decimal track and bounded by neighbouring dates.
                cv2.drawMarker(
                    preview,
                    point,
                    (0, 215, 255),
                    cv2.MARKER_CROSS,
                    7,
                    1,
                    cv2.LINE_AA,
                )
    for item in geometry.virtual_month_anchors:
        month = int(item["month"])
        point = (
            round(
                (
                    vertical_rule_x(
                        geometry, month, float(item["row_center_y"])
                    )
                    + vertical_rule_x(
                        geometry, month + 1, float(item["row_center_y"])
                    )
                )
                / 2
            ),
            round(float(item["row_center_y"])),
        )
        # Hollow cyan diamonds are geometry-only calendar control points; they
        # are deliberately not attached to any printed ink.
        cv2.drawMarker(
            preview,
            point,
            (255, 255, 0),
            cv2.MARKER_DIAMOND,
            8,
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), preview)


def draw_cell_preview(
    image: np.ndarray,
    geometry: PageGeometry,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    preview = image.copy()
    for row in rows:
        color = (0, 0, 255) if row["待确认"] else (0, 170, 0)
        polygon = row.get("单元格多边形")
        if polygon:
            points = np.asarray(
                [[round(x), round(y)] for x, y in polygon], dtype=np.int32
            )
            cv2.polylines(
                preview, [points], True, color, 1, cv2.LINE_AA
            )
        else:
            left, top, right, bottom = row["单元格边界"]
            cv2.rectangle(preview, (left, top), (right, bottom), color, 1)
    cv2.rectangle(
        preview,
        (geometry.table_left, geometry.header_bottom),
        (geometry.table_right, geometry.statistics_top),
        (255, 0, 255),
        2,
    )
    cv2.imwrite(str(output_path), preview)


def write_station_workbook(
    output_dir: Path,
    station: str,
    output_name: str,
    heading: dict[str, Any],
    rows: list[dict[str, Any]],
    geometries: list[PageGeometry],
    month_lengths: list[int],
) -> Path:
    decimal_places = 1
    recognized_texts = [
        normalized_numeric_text(str(row.get("主模型文本", ""))) for row in rows
    ]
    if any(re.search(r"\.\d{2,}", text) for text in recognized_texts):
        decimal_places = 2
    flow_number_format = "0." + "0" * decimal_places
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "合并表"
    headers = ["月", "日", "流量(m³/s)", "平均置信度"]
    sheet.append(headers)
    pending_fill = PatternFill("solid", fgColor="FCE8E6")
    low_fill = PatternFill("solid", fgColor="FFF2CC")
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for row in rows:
        sheet.append(
            [
                row["月"],
                row["日"],
                row["流量(m³/s)"],
                row["平均置信度"],
            ]
        )
        excel_row = sheet.max_row
        sheet.cell(excel_row, 1).number_format = "0"
        sheet.cell(excel_row, 2).number_format = "0"
        sheet.cell(excel_row, 3).number_format = flow_number_format
        sheet.cell(excel_row, 4).number_format = "0.00000000"
        if row["待确认"]:
            for cell in sheet[excel_row]:
                cell.fill = pending_fill
        elif row["平均置信度"] < 0.85:
            for cell in sheet[excel_row]:
                cell.fill = low_fill
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for width, column in zip((8, 8, 16, 18), range(1, 5)):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:D{sheet.max_row}"
    sheet.sheet_view.showGridLines = False
    sheet.row_dimensions[1].height = 25

    matrix = workbook.create_sheet("月日矩阵")
    matrix.merge_cells("A1:M1")
    title_bits = [heading.get("table_number", ""), heading.get("river", ""), station, "逐日平均流量表"]
    matrix["A1"] = "  ".join(bit for bit in title_bits if bit)
    matrix["A1"].font = Font(bold=True, size=16, color="1F2937")
    matrix["A1"].alignment = Alignment(horizontal="center")
    matrix.merge_cells("J2:M2")
    area = heading.get("catchment_area_km2")
    matrix["J2"] = (
        f"集水面积  {area:g} km²，流量  m³/s"
        if isinstance(area, (int, float))
        else "流量  m³/s"
    )
    matrix["J2"].alignment = Alignment(horizontal="right")
    matrix.append(["日 \\ 月", *MONTH_NAMES])
    matrix_header_row = 3
    by_key = {(row["月"], row["日"]): row for row in rows}
    for day in range(1, 32):
        values: list[Any] = [day]
        for month in range(1, 13):
            row = by_key.get((month, day))
            values.append(None if row is None else row["流量(m³/s)"])
        matrix.append(values)
    for cell in matrix[matrix_header_row]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for row_number in range(4, 35):
        for column in range(1, 14):
            cell = matrix.cell(row_number, column)
            cell.alignment = Alignment(horizontal="center")
            if column > 1:
                cell.number_format = flow_number_format
    for column in range(1, 14):
        matrix.column_dimensions[get_column_letter(column)].width = 11
    matrix.column_dimensions["A"].width = 10
    matrix.freeze_panes = "B4"
    matrix.sheet_view.showGridLines = False
    matrix.row_dimensions[1].height = 28

    anchor_sheet = workbook.create_sheet("锚点校核")
    anchor_headers = [
        "页",
        "表区",
        "日",
        "外接框中心Y",
        "墨迹重心Y",
        "支持列数",
        "日列支持",
        "与前行间距",
        "曲线最大偏移",
        "曲线跨列范围",
        "状态",
    ]
    anchor_sheet.append(anchor_headers)
    for geometry in geometries:
        previous: float | None = None
        for day, anchor in enumerate(geometry.row_anchors, start=1):
            gap = None if previous is None else anchor.center - previous
            curve = (
                geometry.row_curve_centers[day - 1]
                if day - 1 < len(geometry.row_curve_centers)
                else [anchor.center]
            )
            maximum_deviation = max(abs(value - anchor.center) for value in curve)
            cross_column_range = max(curve) - min(curve)
            status = (
                "通过"
                if anchor.support_columns >= 5 and anchor.day_column_support
                else "待确认"
            )
            anchor_sheet.append(
                [
                    geometry.page,
                    geometry.table_index,
                    day,
                    anchor.center,
                    anchor.ink_centroid,
                    anchor.support_columns,
                    "是" if anchor.day_column_support else "否",
                    gap,
                    maximum_deviation,
                    cross_column_range,
                    status,
                ]
            )
            previous = anchor.center
    for cell in anchor_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for row_number in range(2, anchor_sheet.max_row + 1):
        for column in (4, 5, 8, 9, 10):
            anchor_sheet.cell(row_number, column).number_format = "0.000"
    for column, width in enumerate(
        (8, 8, 8, 16, 16, 12, 12, 16, 16, 16, 12), start=1
    ):
        anchor_sheet.column_dimensions[get_column_letter(column)].width = width
    anchor_sheet.freeze_panes = "A2"
    anchor_sheet.auto_filter.ref = f"A1:K{anchor_sheet.max_row}"
    anchor_sheet.sheet_view.showGridLines = False

    pending = workbook.create_sheet("待确认项")
    pending_headers = [
        "输出行",
        "页",
        "月",
        "日",
        "建议值",
        "主模型文本",
        "辅助模型文本",
        "平均置信度",
        "原因",
        "样本ID",
        "图片",
    ]
    pending.append(pending_headers)
    for output_row, row in enumerate(rows, start=1):
        if not row["待确认"]:
            continue
        pending.append(
            [
                output_row,
                row["页"],
                row["月"],
                row["日"],
                row["流量(m³/s)"],
                excel_literal_text(row["主模型文本"]),
                excel_literal_text(row["辅助模型文本"]),
                row["平均置信度"],
                row["备注"],
                row["样本ID"],
                row["图片"],
            ]
        )
        pending.cell(pending.max_row, 11).hyperlink = row["图片"]
        pending.cell(pending.max_row, 11).style = "Hyperlink"
    for cell in pending[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
    for column, width in enumerate(
        (10, 8, 8, 8, 14, 18, 18, 16, 52, 30, 44), start=1
    ):
        pending.column_dimensions[get_column_letter(column)].width = width
    pending.freeze_panes = "A2"
    pending.auto_filter.ref = f"A1:K{max(1, pending.max_row)}"
    pending.sheet_view.showGridLines = False

    workbook_path = output_dir / f"{safe_filename(output_name)}.xlsx"
    workbook.save(workbook_path)
    return workbook_path


def station_output_name(station: str, occurrence: int, total: int) -> str:
    """Return a stable filename label while preserving the Chinese station name."""

    if total <= 1:
        return station
    return f"{station}_{occurrence:02d}"


def write_training_review_workbook(
    samples: list[dict[str, Any]], training_dir: Path
) -> Path:
    workbook = Workbook()
    instructions = workbook.active
    instructions.title = "使用说明"
    instructions.append(["逐日流量OCR训练样本校正说明"])
    instructions.append(["1", "打开“标签校正”工作表。"])
    instructions.append(["2", "点击图片文件名查看实际识别的流量单元格。"])
    instructions.append(["3", "只修改“正确标签”和“纳入训练”两列。"])
    instructions.append(["4", "正确标签必须与图片完全一致，并保留小数点。"])
    instructions.append(["5", "确认无误请选择“是”；无法判断请选择“待确认”。"])
    instructions.append(["6", "程序使用多列墨迹锚点重建31个日序，空白格不拟合坐标。"])
    instructions.append(["7", "可用 --corrected-labels 指向校正后的工作簿重新输出。"])
    instructions.column_dimensions["A"].width = 18
    instructions.column_dimensions["B"].width = 100
    instructions.merge_cells("A1:B1")
    instructions["A1"].font = Font(bold=True, size=15, color="FFFFFF")
    instructions["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    instructions.row_dimensions[1].height = 28
    for row in range(2, instructions.max_row + 1):
        instructions.cell(row, 1).font = Font(bold=True, color="1F4E78")
        instructions.cell(row, 2).alignment = Alignment(wrap_text=True)
        instructions.row_dimensions[row].height = 26
    instructions.sheet_view.showGridLines = False

    sheet = workbook.create_sheet("标签校正")
    headers = [
        "样本ID",
        "页",
        "表区",
        "行序号",
        "月",
        "日",
        "字段",
        "主模型原文",
        "主模型置信度",
        "辅助模型原文",
        "辅助模型置信度",
        "规范化标签",
        "正确标签",
        "纳入训练",
        "图片文件",
        "备注",
    ]
    sheet.append(headers)
    for sample in samples:
        sheet.append(
            [
                sample["sample_id"],
                sample["page"],
                sample["panel"],
                sample["row"],
                sample["month"],
                sample["day"],
                sample["column"],
                excel_literal_text(sample["primary_text"]),
                sample["primary_confidence"],
                excel_literal_text(sample["secondary_text"]),
                sample["secondary_confidence"],
                excel_literal_text(sample["normalized_label"]),
                excel_literal_text(sample["correct_label"]),
                sample["include"],
                sample["image"],
                excel_literal_text(sample.get("note", "")),
            ]
        )
        image_cell = sheet.cell(sheet.max_row, 15)
        image_cell.hyperlink = sample["image"]
        image_cell.style = "Hyperlink"

    header_fill = PatternFill("solid", fgColor="1F4E78")
    low_fill = PatternFill("solid", fgColor="FFF2CC")
    review_fill = PatternFill("solid", fgColor="FCE4D6")
    editable_fill = PatternFill("solid", fgColor="E2F0D9")
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for row in range(2, sheet.max_row + 1):
        sheet.cell(row, 9).number_format = "0.00000000"
        sheet.cell(row, 11).number_format = "0.00000000"
        sheet.cell(row, 13).number_format = "@"
        sheet.cell(row, 13).fill = editable_fill
        sheet.cell(row, 14).fill = editable_fill
        confidence = min(
            sheet.cell(row, 9).value or 0.0,
            sheet.cell(row, 11).value or 1.0,
        )
        if confidence < 0.85:
            for column in range(1, 13):
                sheet.cell(row, column).fill = low_fill
        if sheet.cell(row, 14).value == "待确认":
            sheet.cell(row, 14).fill = review_fill
    validation = DataValidation(
        type="list", formula1='"是,否,待确认"', allow_blank=False
    )
    sheet.add_data_validation(validation)
    validation.add(f"N2:N{sheet.max_row}")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:P{sheet.max_row}"
    sheet.row_dimensions[1].height = 26
    widths = [30, 8, 8, 10, 8, 8, 18, 18, 15, 18, 15, 18, 18, 14, 48, 52]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.sheet_view.showGridLines = False
    path = training_dir / "labels_review.xlsx"
    workbook.save(path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "识别按日×月排版的逐日平均流量扫描表；"
            "通过多列墨迹锚点重建31个日序，并排除下方统计区。"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", type=Path, help="输入PDF文件")
    parser.add_argument("-o", "--output", type=Path, help="输出目录")
    parser.add_argument("--device", default="gpu:0", help="Paddle推理设备")
    parser.add_argument("--render-width", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--low-confidence", type=float, default=0.85)
    parser.add_argument(
        "--primary-model-name",
        default="PP-OCRv5_server_rec",
        help="标题和流量主识别模型",
    )
    parser.add_argument(
        "--secondary-model-name",
        default="en_PP-OCRv5_mobile_rec",
        help="流量辅助数字模型",
    )
    parser.add_argument(
        "--disable-secondary-model",
        action="store_true",
        help="禁用辅助数字模型",
    )
    parser.add_argument(
        "--corrected-labels",
        type=Path,
        help="可选的labels_review校正工作簿",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"输入PDF不存在：{pdf_path}")
    script_dir = Path(__file__).resolve().parent
    output_dir = (
        args.output.expanduser().resolve()
        if args.output
        else script_dir / "output_dailyflowrate" / f"{pdf_path.stem}_result"
    )
    structure_dir = output_dir / "structure"
    previews_dir = output_dir / "previews"
    training_dir = output_dir / "training_samples"
    training_images_dir = training_dir / "images"
    for directory in (
        output_dir,
        structure_dir,
        previews_dir,
        training_images_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    verify_device(args.device)
    log("[1/4] 初始化印刷体文字主模型")
    primary_recognizer = TextRecognition(
        model_name=args.primary_model_name,
        device=args.device,
        engine="paddle",
    )
    secondary_recognizer: TextRecognition | None = None
    if not args.disable_secondary_model:
        log("[1/4] 初始化印刷数字辅助模型")
        secondary_recognizer = TextRecognition(
            model_name=args.secondary_model_name,
            device=args.device,
            engine="paddle",
        )

    document = pdfium.PdfDocument(str(pdf_path))
    page_count = len(document)
    document.close()
    if page_count < 1:
        raise RuntimeError("PDF没有页面。")

    all_rows: list[dict[str, Any]] = []
    all_samples: list[dict[str, Any]] = []
    dataset_groups: dict[str, dict[str, Any]] = {}

    for page_index in range(page_count):
        page_number = page_index + 1
        log(f"[2/4] 渲染并定位第{page_number}/{page_count}页")
        source_image = render_pdf_page(pdf_path, page_index, args.render_width)
        regions = detect_table_regions(source_image, page_number)
        if len(regions) > 1 and args.render_width < 2200:
            log(
                f"[2/4] 第{page_number}页为照片多表页，"
                "自动提高到2200像素宽以保留小号数字"
            )
            source_image = render_pdf_page(pdf_path, page_index, 2200)
            regions = detect_table_regions(source_image, page_number)
        cv2.imwrite(
            str(structure_dir / f"page_{page_number:03d}_source.png"), source_image
        )
        work_items: list[tuple[np.ndarray, PageGeometry, dict[str, Any], str]] = []
        if len(regions) == 1:
            try:
                geometry = detect_page_geometry(source_image, page_number, 1)
                heading = recognize_page_heading(
                    source_image, geometry, primary_recognizer
                )
                work_items.append((source_image, geometry, heading, "original"))
            except RuntimeError:
                pass
        if not work_items:
            if not regions:
                raise RuntimeError(
                    f"第{page_number}页没有检测到由14条竖线组成的逐日流量表。"
                )
            for region in regions:
                rectified = rectify_table_region(source_image, region)
                region.image = rectified
                geometry = detect_rectified_geometry(rectified, region)
                heading = recognize_region_heading(
                    source_image, region, primary_recognizer
                )
                work_items.append((rectified, geometry, heading, "photo-dewarped"))

        log(f"[2/4] 第{page_number}页检测到{len(work_items)}张站表")
        for table_image, geometry, heading, geometry_mode in work_items:
            table_index = geometry.table_index
            log(
                f"[2/4] 第{page_number}页表{table_index}："
                f"站点={heading['station']}，候选锚点="
                f"{len(geometry.candidate_anchors)}，最终锚点="
                f"{len(geometry.row_anchors)}"
            )
            rows, samples, month_lengths = recognize_daily_cells(
                table_image,
                geometry,
                heading,
                primary_recognizer,
                secondary_recognizer,
                training_images_dir,
                args.batch_size,
                args.low_confidence,
            )
            geometry_json = geometry_to_json(geometry)
            geometry_json["heading"] = heading
            geometry_json["month_lengths"] = month_lengths
            geometry_json["geometry_mode"] = geometry_mode
            geometry_path = structure_dir / (
                f"page_{page_number:03d}_table_{table_index:02d}_geometry.json"
            )
            geometry_path.write_text(
                json.dumps(geometry_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cv2.imwrite(
                str(
                    structure_dir
                    / f"page_{page_number:03d}_table_{table_index:02d}_rectified.png"
                ),
                table_image,
            )
            draw_anchor_preview(
                table_image,
                geometry,
                previews_dir
                / f"page_{page_number:03d}_table_{table_index:02d}_anchors.png",
            )
            draw_cell_preview(
                table_image,
                geometry,
                rows,
                previews_dir
                / f"page_{page_number:03d}_table_{table_index:02d}_cells.png",
            )
            table_crop = table_image[
                max(0, geometry.table_top) : min(
                    geometry.height, geometry.statistics_top + 2
                ),
                max(0, geometry.table_left) : min(
                    geometry.width, geometry.table_right + 1
                ),
            ]
            cv2.imwrite(
                str(
                    previews_dir
                    / f"page_{page_number:03d}_table_{table_index:02d}_table.png"
                ),
                table_crop,
            )
            entry = {
                "page": page_number,
                "table_index": table_index,
                "station": heading["station"],
                "row_count": len(rows),
                "month_lengths": month_lengths,
                "pending_count": sum(1 for row in rows if row["待确认"]),
                "geometry_mode": geometry_mode,
                "geometry": str(geometry_path),
            }
            # One detected table is one annual dataset.  Repeated Chinese
            # station names are deliberately not merged because adjacent PDF
            # pages can represent different years whose year OCR is deferred.
            dataset_key = f"page_{page_number:03d}_table_{table_index:02d}"
            if dataset_key in dataset_groups:
                raise RuntimeError(f"年度表键重复：{dataset_key}")
            group = {
                "station": heading["station"],
                "heading": heading,
                "rows": [],
                "samples": [],
                "geometries": [],
                "month_lengths": month_lengths,
                "tables": [],
            }
            dataset_groups[dataset_key] = group
            group["rows"].extend(rows)
            group["samples"].extend(samples)
            group["geometries"].append(geometry)
            group["tables"].append(entry)
            all_rows.extend(rows)
            all_samples.extend(samples)

    # The scan is traversed by day then month to preserve row-anchor locality.
    # Final hydrological records must be chronological by month then day.
    all_rows.sort(
        key=lambda row: (row["页"], row["表区"], row["月"], row["日"])
    )

    if args.corrected_labels:
        corrected_path = args.corrected_labels.expanduser().resolve()
        if not corrected_path.is_file():
            raise FileNotFoundError(f"标签校正表不存在：{corrected_path}")
        applied = apply_corrected_labels(all_samples, corrected_path)
        synchronize_rows_from_samples(all_rows, all_samples)
        log(f"[3/4] 已应用{applied}处人工校正标签")

    station_totals = Counter(
        str(group["station"]) for group in dataset_groups.values()
    )
    station_seen: Counter[str] = Counter()
    manifest_stations: list[dict[str, Any]] = []
    for dataset_key, group in dataset_groups.items():
        station = str(group["station"])
        station_seen[station] += 1
        occurrence = station_seen[station]
        output_name = station_output_name(
            station, occurrence, station_totals[station]
        )
        group["rows"].sort(
            key=lambda row: (row["页"], row["表区"], row["月"], row["日"])
        )
        workbook_path = write_station_workbook(
            output_dir,
            station,
            output_name,
            group["heading"],
            group["rows"],
            group["geometries"],
            group["month_lengths"],
        )
        station_json_path = output_dir / f"{safe_filename(output_name)}.json"
        station_json_path.write_text(
            json.dumps(
                {
                    "input_pdf": str(pdf_path),
                    "station": station,
                    "dataset_key": dataset_key,
                    "output_name": output_name,
                    "station_occurrence": occurrence,
                    "station_occurrence_count": station_totals[station],
                    "heading": group["heading"],
                    "row_count": len(group["rows"]),
                    "pending_count": sum(
                        1 for row in group["rows"] if row["待确认"]
                    ),
                    "tables": group["tables"],
                    "rows": group["rows"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest_stations.append(
            {
                "station": station,
                "dataset_key": dataset_key,
                "output_name": output_name,
                "station_occurrence": occurrence,
                "station_occurrence_count": station_totals[station],
                "row_count": len(group["rows"]),
                "pending_count": sum(
                    1 for row in group["rows"] if row["待确认"]
                ),
                "excel": str(workbook_path),
                "json": str(station_json_path),
            }
        )
    review_path = write_training_review_workbook(all_samples, training_dir)
    manifest_path = output_dir / f"{pdf_path.stem}_stations.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_pdf": str(pdf_path),
                "station_count": len(station_totals),
                "dataset_count": len(dataset_groups),
                "stations": manifest_stations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(
        f"[4/4] 完成：{len(station_totals)}个站点，"
        f"{len(dataset_groups)}张年度表，"
        f"{len(all_rows)}条逐日流量"
    )
    for item in manifest_stations:
        log(f"[完成] {item['output_name']} Excel：{item['excel']}")
        log(f"[完成] {item['output_name']} JSON：{item['json']}")
    log(f"[完成] 锚点/分格预览：{previews_dir}")
    log(f"[完成] 标签校正表：{review_path}")
    log(f"[完成] 站点清单：{manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("已取消。")
        raise SystemExit(130)
