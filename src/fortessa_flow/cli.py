#!/usr/bin/env python3
"""Read-only, configuration-driven FCS 3.0 analysis workflow."""

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np


CHANNELS = {
    "fsc_a": "FSC-A",
    "fsc_h": "FSC-H",
    "ssc_a": "SSC-A",
    "dapi": "V 450/50-A",
    "af488": "Blue 530/30-A",
    "af568": "YG 586/15-A",
    "time": "Time",
}
FLUORESCENCE = [CHANNELS["dapi"], CHANNELS["af488"], CHANNELS["af568"]]
LABELS = {"dapi": "DAPI", "af488": "Marker 1 AF488", "af568": "Marker 2 AF568"}
CONTROL_GROUPS = {
    "background": "Secondary-only background",
    "dapi": "DAPI-only",
    "af488": "Marker 1 single-primary control",
    "af568": "Marker 2 single-primary control",
}
SETTINGS = {
    "time_trim_fraction": 0.005,
    "scatter_mad_limit": 4.0,
    "singlet_mad_limit": 3.5,
    "background_percentile": 0.99,
    "maximum_accepted_coefficient": 0.2,
    "maximum_replicate_ratio": 2.0,
    "dapi_cofactor": 1000.0,
    "af488_cofactor": 100.0,
    "af568_cofactor": 100.0,
    "require_matching_voltages": True,
}
ANALYSIS_TITLE = "Three-colour flow-cytometry analysis"
COMPENSATION_PAIRS = [("af488", "af568"), ("af568", "af488")]


def configure(config):
    global CHANNELS, FLUORESCENCE, LABELS, CONTROL_GROUPS, SETTINGS, ANALYSIS_TITLE, COMPENSATION_PAIRS
    channel_config = config["channels"]
    CHANNELS = {
        "fsc_a": channel_config["fsc_a"],
        "fsc_h": channel_config["fsc_h"],
        "ssc_a": channel_config["ssc_a"],
        "dapi": channel_config["dapi"],
        "af488": channel_config["marker_1"],
        "af568": channel_config["marker_2"],
        "time": channel_config["time"],
    }
    FLUORESCENCE = [CHANNELS["dapi"], CHANNELS["af488"], CHANNELS["af568"]]
    label_config = config.get("labels", {})
    LABELS = {
        "dapi": label_config.get("dapi", "DAPI"),
        "af488": label_config.get("marker_1", "Marker 1"),
        "af568": label_config.get("marker_2", "Marker 2"),
    }
    group_config = config["control_groups"]
    CONTROL_GROUPS = {
        "background": group_config["background"],
        "dapi": group_config["dapi"],
        "af488": group_config["marker_1"],
        "af568": group_config["marker_2"],
    }
    gating = config.get("gating", {})
    thresholds = config.get("thresholds", {})
    compensation = config.get("compensation", {})
    transforms = config.get("transforms", {})
    qc = config.get("qc", {})
    SETTINGS = {
        "time_trim_fraction": float(gating.get("time_trim_fraction", 0.005)),
        "scatter_mad_limit": float(gating.get("scatter_mad_limit", 4.0)),
        "singlet_mad_limit": float(gating.get("singlet_mad_limit", 3.5)),
        "background_percentile": float(thresholds.get("background_percentile", 0.99)),
        "maximum_accepted_coefficient": float(compensation.get("maximum_accepted_coefficient", 0.2)),
        "maximum_replicate_ratio": float(compensation.get("maximum_replicate_ratio", 2.0)),
        "dapi_cofactor": float(transforms.get("dapi_cofactor", 1000)),
        "af488_cofactor": float(transforms.get("marker_1_cofactor", 100)),
        "af568_cofactor": float(transforms.get("marker_2_cofactor", 100)),
        "require_matching_voltages": bool(qc.get("require_matching_voltages", True)),
    }
    key_alias = {"marker_1": "af488", "marker_2": "af568", "dapi": "dapi"}
    COMPENSATION_PAIRS = [
        (key_alias[pair["source"]], key_alias[pair["target"]])
        for pair in compensation.get("pairs", [])
    ]
    ANALYSIS_TITLE = config.get("analysis_title", ANALYSIS_TITLE)


def validate_config(config):
    required_sections = ("paths", "channels", "control_groups")
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration is missing required sections: {missing_sections}")
    required_paths = ("input_dir", "manifest", "output_dir")
    missing_paths = [key for key in required_paths if not config["paths"].get(key)]
    if missing_paths:
        raise ValueError(f"Configuration paths are missing: {missing_paths}")
    mode = config.get("thresholds", {}).get("mode", "background_percentile")
    if mode not in {"background_percentile", "flowjo_calibrated"}:
        raise ValueError(f"Unsupported threshold mode: {mode}")
    if mode == "flowjo_calibrated" and not config["paths"].get("flowjo_targets"):
        raise ValueError("flowjo_calibrated mode requires paths.flowjo_targets")
    percentile = float(config.get("thresholds", {}).get("background_percentile", 0.99))
    if not 0 < percentile < 1:
        raise ValueError("thresholds.background_percentile must be between 0 and 1")
    pairs = config.get("compensation", {}).get("pairs", [])
    if not pairs:
        raise ValueError("compensation.pairs must contain at least one source/target pair")
    valid_channels = {"dapi", "marker_1", "marker_2"}
    for pair in pairs:
        if pair.get("source") not in valid_channels or pair.get("target") not in valid_channels:
            raise ValueError(f"Invalid compensation pair: {pair}")
        if pair["source"] == pair["target"]:
            raise ValueError(f"Compensation source and target must differ: {pair}")


def resolve_config_path(config_path, value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def as_int(value, default=None):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_text(raw):
    delimiter = chr(raw[0])
    text = raw[1:].decode("latin-1", errors="replace")
    marker = "\0DELIMITER\0"
    fields = [x.replace(marker, delimiter) for x in text.replace(delimiter * 2, marker).split(delimiter)]
    if fields and fields[-1] == "":
        fields.pop()
    return {fields[i].upper(): fields[i + 1] for i in range(0, len(fields) - 1, 2)}


def meta_value(metadata, *keys):
    for key in keys:
        if key.upper() in metadata:
            return metadata[key.upper()]
    return ""


def read_fcs(path):
    with path.open("rb") as handle:
        header = handle.read(58)
        if len(header) < 58 or not header.startswith(b"FCS3.0"):
            raise ValueError(f"{path.name}: not an FCS 3.0 file")
        text_start = as_int(header[10:18].decode("ascii", errors="ignore"))
        text_end = as_int(header[18:26].decode("ascii", errors="ignore"))
        handle.seek(text_start)
        metadata = parse_text(handle.read(text_end - text_start + 1))

        data_start = as_int(meta_value(metadata, "$BEGINDATA"), as_int(header[26:34].decode("ascii", errors="ignore")))
        data_end = as_int(meta_value(metadata, "$ENDDATA"), as_int(header[34:42].decode("ascii", errors="ignore")))
        event_count = as_int(meta_value(metadata, "$TOT"), 0)
        parameter_count = as_int(meta_value(metadata, "$PAR"), 0)
        datatype = meta_value(metadata, "$DATATYPE").upper()
        byte_order = meta_value(metadata, "$BYTEORD")
        endian = "<" if byte_order.startswith("1,2") else ">"

        names, voltages, ranges, bits = [], [], [], []
        for index in range(1, parameter_count + 1):
            names.append(meta_value(metadata, f"$P{index}N"))
            voltages.append(meta_value(metadata, f"$P{index}V"))
            ranges.append(as_int(meta_value(metadata, f"$P{index}R"), 0))
            bits.append(as_int(meta_value(metadata, f"$P{index}B"), 0))

        handle.seek(data_start)
        raw = handle.read(data_end - data_start + 1)

    if datatype == "F":
        values = np.frombuffer(raw, dtype=np.dtype(endian + "f4"), count=event_count * parameter_count)
    elif datatype == "D":
        values = np.frombuffer(raw, dtype=np.dtype(endian + "f8"), count=event_count * parameter_count)
    elif datatype == "I" and len(set(bits)) == 1 and bits[0] in (8, 16, 32, 64):
        values = np.frombuffer(raw, dtype=np.dtype(endian + f"u{bits[0] // 8}"), count=event_count * parameter_count)
    else:
        raise ValueError(f"{path.name}: unsupported DATA encoding {datatype}/{bits}")

    if values.size != event_count * parameter_count:
        raise ValueError(f"{path.name}: DATA size mismatch")
    return {
        "metadata": metadata,
        "names": names,
        "voltages": voltages,
        "ranges": ranges,
        "data": values.reshape(event_count, parameter_count).astype(np.float64, copy=False),
    }


def robust_mad(values):
    median = np.median(values)
    return median, 1.4826 * np.median(np.abs(values - median))


def build_gate(fcs):
    index = {name: i for i, name in enumerate(fcs["names"])}
    data = fcs["data"]
    required = [index[CHANNELS[k]] for k in ("fsc_a", "fsc_h", "ssc_a", "time")]
    mask = np.all(np.isfinite(data[:, required]), axis=1)
    mask &= data[:, index[CHANNELS["fsc_a"]]] > 0
    mask &= data[:, index[CHANNELS["fsc_h"]]] > 0
    mask &= data[:, index[CHANNELS["ssc_a"]]] > 0

    time = data[:, index[CHANNELS["time"]]]
    trim = SETTINGS["time_trim_fraction"]
    t_low, t_high = np.quantile(time[mask], [trim, 1 - trim])
    mask &= (time >= t_low) & (time <= t_high)

    log_fsc = np.log10(np.clip(data[:, index[CHANNELS["fsc_a"]]], 1, None))
    log_ssc = np.log10(np.clip(data[:, index[CHANNELS["ssc_a"]]], 1, None))
    for values in (log_fsc, log_ssc):
        centre, scale = robust_mad(values[mask])
        if scale > 0:
            mask &= np.abs(values - centre) <= SETTINGS["scatter_mad_limit"] * scale

    ratio = np.log10(np.clip(data[:, index[CHANNELS["fsc_h"]]], 1, None) /
                     np.clip(data[:, index[CHANNELS["fsc_a"]]], 1, None))
    centre, scale = robust_mad(ratio[mask])
    if scale > 0:
        mask &= np.abs(ratio - centre) <= SETTINGS["singlet_mad_limit"] * scale
    return mask


def load_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "filename", "role", "control_group", "replicate_group",
            "include_for_compensation", "include_for_analysis", "notes",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
        rows = list(reader)
    filenames = [row["filename"] for row in rows]
    duplicates = sorted({filename for filename in filenames if filenames.count(filename) > 1})
    if duplicates:
        raise ValueError(f"Manifest contains duplicate filenames: {duplicates}")
    for row in rows:
        for field in ("include_for_compensation", "include_for_analysis"):
            if row[field] not in {"Yes", "No"}:
                raise ValueError(f"{row['filename']}: {field} must be Yes or No")
    return {row["filename"]: row for row in rows}


def percentile_summary(values):
    q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return dict(zip(("p05", "p25", "median", "p75", "p95", "p99"), map(float, q)))


def slope_with_bootstrap(x, y, seed, iterations=200):
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    low, high = np.quantile(x, [0.02, 0.98])
    keep = (x >= low) & (x <= high)
    x, y = x[keep], y[keep]
    if x.size < 100 or np.var(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    if x.size > 2500:
        subset = rng.choice(x.size, size=2500, replace=False)
        x, y = x[subset], y[subset]
    slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    corr = float(np.corrcoef(x, y)[0, 1])
    estimates = []
    for _ in range(iterations):
        idx = rng.integers(0, x.size, x.size)
        xb, yb = x[idx], y[idx]
        variance = np.var(xb, ddof=1)
        if variance > 0:
            estimates.append(np.cov(xb, yb, ddof=1)[0, 1] / variance)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return slope, float(lower), float(upper), corr


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def transformed(values, cofactor):
    return np.arcsinh(values / cofactor)


def load_flowjo_targets(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["filename"]: {
                "marker_2_only_pct": float(row["marker_2_only_pct"]),
                "marker_1_only_pct": float(row["marker_1_only_pct"]),
                "double_positive_pct": float(row["double_positive_pct"]),
            }
            for row in csv.DictReader(handle)
        }


def calibrate_quadrant_thresholds(datasets, targets, channel_position):
    filenames = list(targets)
    missing = [filename for filename in filenames if filename not in datasets]
    if missing:
        raise ValueError(f"FlowJo calibration files missing from analysis: {missing}")

    x_values = [datasets[filename]["compensated"][datasets[filename]["gate"], channel_position[CHANNELS["af488"]]] for filename in filenames]
    y_values = [datasets[filename]["compensated"][datasets[filename]["gate"], channel_position[CHANNELS["af568"]]] for filename in filenames]
    pooled_x = np.concatenate(x_values)
    pooled_y = np.concatenate(y_values)
    x_candidates = np.unique(np.quantile(pooled_x, np.linspace(0.80, 0.9995, 600)))
    y_candidates = np.unique(np.quantile(pooled_y, np.linspace(0.10, 0.95, 600)))

    x_predictions = np.array([
        100 * (values.size - np.searchsorted(np.sort(values), x_candidates, side="right")) / values.size
        for values in x_values
    ])
    y_predictions = np.array([
        100 * (values.size - np.searchsorted(np.sort(values), y_candidates, side="right")) / values.size
        for values in y_values
    ])
    target_x = np.array([targets[name]["marker_1_only_pct"] + targets[name]["double_positive_pct"] for name in filenames])
    target_y = np.array([targets[name]["marker_2_only_pct"] + targets[name]["double_positive_pct"] for name in filenames])
    top_x = np.argsort(np.mean((x_predictions - target_x[:, None]) ** 2, axis=0))[:80]
    top_y = np.argsort(np.mean((y_predictions - target_y[:, None]) ** 2, axis=0))[:80]
    selected_x = x_candidates[top_x]
    selected_y = y_candidates[top_y]

    total_error = np.zeros((len(selected_x), len(selected_y)))
    per_sample = {}
    for sample_index, filename in enumerate(filenames):
        x = x_values[sample_index]
        y = y_values[sample_index]
        dp = np.empty((len(selected_x), len(selected_y)))
        for x_index, threshold_x in enumerate(selected_x):
            positive_y = np.sort(y[x > threshold_x])
            dp[x_index, :] = 100 * (positive_y.size - np.searchsorted(positive_y, selected_y, side="right")) / x.size
        x_total = x_predictions[sample_index, top_x][:, None]
        y_total = y_predictions[sample_index, top_y][None, :]
        gh_only = x_total - dp
        pk_only = y_total - dp
        target = targets[filename]
        error = (
            (pk_only - target["marker_2_only_pct"]) ** 2
            + (gh_only - target["marker_1_only_pct"]) ** 2
            + (dp - target["double_positive_pct"]) ** 2
        )
        total_error += error
        per_sample[filename] = (pk_only, gh_only, dp)

    best_x_index, best_y_index = np.unravel_index(np.argmin(total_error), total_error.shape)
    threshold_x = float(selected_x[best_x_index])
    threshold_y = float(selected_y[best_y_index])
    fit_rows = []
    for filename in filenames:
        pk_only, gh_only, dp = per_sample[filename]
        target = targets[filename]
        predicted = {
            "marker_2_only_pct": float(pk_only[best_x_index, best_y_index]),
            "marker_1_only_pct": float(gh_only[best_x_index, best_y_index]),
            "double_positive_pct": float(dp[best_x_index, best_y_index]),
        }
        fit_rows.append({
            "filename": filename,
            "flowjo_marker_2_only_pct": target["marker_2_only_pct"],
            "calibrated_marker_2_only_pct": predicted["marker_2_only_pct"],
            "marker_2_difference_points": predicted["marker_2_only_pct"] - target["marker_2_only_pct"],
            "flowjo_marker_1_only_pct": target["marker_1_only_pct"],
            "calibrated_marker_1_only_pct": predicted["marker_1_only_pct"],
            "marker_1_difference_points": predicted["marker_1_only_pct"] - target["marker_1_only_pct"],
            "flowjo_double_positive_pct": target["double_positive_pct"],
            "calibrated_double_positive_pct": predicted["double_positive_pct"],
            "double_positive_difference_points": predicted["double_positive_pct"] - target["double_positive_pct"],
        })
    return {CHANNELS["af488"]: threshold_x, CHANNELS["af568"]: threshold_y}, fit_rows


def save_svg(path, body, width, height):
    document = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;fill:#17202a}}.axis{{stroke:#667788;stroke-width:1}}.grid{{stroke:#e1e7ec;stroke-width:1}}</style>{body}</svg>'''
    path.write_text(document, encoding="utf-8")


def bar_panel_svg(sample_rows, labels):
    width, height = 1100, 720
    left, right = 85, 30
    panel_height = 260
    parts = [f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-weight="bold">Sample marker summary</text>']
    for panel, (label, colour) in enumerate((("af488", "#2878B5"), ("af568", "#D45A3A"))):
        top = 55 + panel * 330
        bottom = top + panel_height
        values = np.array([row[f"{label}_net_median"] for row in sample_rows], dtype=float)
        positives = [row[f"{label}_positive_pct"] for row in sample_rows]
        ymax = max(float(np.max(values)), 1.0) * 1.15
        xstep = (width - left - right) / max(len(labels), 1)
        display_label = LABELS[label]
        parts.append(f'<text x="{left}" y="{top-10}" font-size="16" font-weight="bold">{html.escape(display_label)} median minus matched background</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        for tick in range(5):
            value = ymax * tick / 4
            y = bottom - panel_height * tick / 4
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11">{value:.0f}</text>')
        for i, (value, pct, text_label) in enumerate(zip(values, positives, labels)):
            x = left + i * xstep + xstep * 0.18
            bar_width = xstep * 0.64
            bar_height = max(0, value) / ymax * panel_height
            y = bottom - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colour}" opacity="0.86"/>')
            parts.append(f'<text x="{x+bar_width/2:.1f}" y="{max(top+12,y-5):.1f}" text-anchor="middle" font-size="10">{pct:.1f}%</text>')
            parts.append(f'<text x="{x+bar_width/2:.1f}" y="{bottom+18}" text-anchor="end" font-size="10" transform="rotate(-35 {x+bar_width/2:.1f} {bottom+18})">{html.escape(text_label)}</text>')
    return "".join(parts), width, height


def distribution_svg(datasets, channel_values):
    width, height = 1200, 420
    groups = [
        (CONTROL_GROUPS["background"], "#666666"), (CONTROL_GROUPS["dapi"], "#7A5195"),
        (CONTROL_GROUPS["af488"], "#2878B5"),
        (CONTROL_GROUPS["af568"], "#D45A3A")
    ]
    cofactors = {
        CHANNELS["dapi"]: SETTINGS["dapi_cofactor"],
        CHANNELS["af488"]: SETTINGS["af488_cofactor"],
        CHANNELS["af568"]: SETTINGS["af568_cofactor"],
    }
    parts = ['<text x="600" y="25" text-anchor="middle" font-size="20" font-weight="bold">Control distributions</text>']
    panel_width, top, bottom = 370, 55, 340
    for panel, channel in enumerate(FLUORESCENCE):
        left = 45 + panel * 395
        all_series = []
        for group, colour in groups:
            controls = [d for d in datasets.values() if d["manifest"]["control_group"] == group]
            if controls:
                values = transformed(np.concatenate([channel_values(d, channel) for d in controls]), cofactors[channel])
                all_series.append((group, colour, values))
        xmin = min(float(np.quantile(v, 0.001)) for _, _, v in all_series)
        xmax = max(float(np.quantile(v, 0.999)) for _, _, v in all_series)
        edges = np.linspace(xmin, xmax, 90)
        histograms = [(g, c, np.histogram(v, bins=edges, density=True)[0]) for g, c, v in all_series]
        ymax = max(float(h.max()) for _, _, h in histograms) * 1.05
        parts.append(f'<text x="{left+panel_width/2}" y="45" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(channel)}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left+panel_width}" y2="{bottom}"/><line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        centres = (edges[:-1] + edges[1:]) / 2
        for group, colour, hist_values in histograms:
            points = []
            for xv, yv in zip(centres, hist_values):
                x = left + (xv - xmin) / (xmax - xmin) * panel_width
                y = bottom - yv / ymax * (bottom - top)
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colour}" stroke-width="1.6"/>')
        parts.append(f'<text x="{left+panel_width/2}" y="365" text-anchor="middle" font-size="11">asinh(signal/{cofactors[channel]})</text>')
    for i, (group, colour) in enumerate(groups):
        x = 45 + (i % 2) * 350
        y = 390 + (i // 2) * 18
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+24}" y2="{y}" stroke="{colour}" stroke-width="2"/><text x="{x+30}" y="{y+4}" font-size="10">{html.escape(group)}</text>')
    return "".join(parts), width, height


def scatter_svg(datasets, channel_values):
    width, height = 1000, 430
    specifications = [
        (CONTROL_GROUPS["af488"], CHANNELS["af488"], CHANNELS["af568"]),
        (CONTROL_GROUPS["af568"], CHANNELS["af568"], CHANNELS["af488"]),
    ]
    parts = ['<text x="500" y="24" text-anchor="middle" font-size="20" font-weight="bold">Cross-channel control diagnostics</text>']
    colours = ["#2878B5", "#D45A3A"]
    for panel, (group, xchannel, ychannel) in enumerate(specifications):
        left, top, panel_width, panel_height = 60 + panel * 490, 60, 410, 300
        controls = [d for d in datasets.values() if d["manifest"]["control_group"] == group]
        series = []
        for dataset in controls:
            xv = transformed(channel_values(dataset, xchannel), 100)
            yv = transformed(channel_values(dataset, ychannel), 100)
            select = np.linspace(0, len(xv) - 1, min(800, len(xv)), dtype=int)
            series.append((xv[select], yv[select]))
        xmin = min(float(np.quantile(x, 0.01)) for x, _ in series); xmax = max(float(np.quantile(x, 0.99)) for x, _ in series)
        ymin = min(float(np.quantile(y, 0.01)) for _, y in series); ymax = max(float(np.quantile(y, 0.99)) for _, y in series)
        parts.append(f'<text x="{left+panel_width/2}" y="45" text-anchor="middle" font-size="13" font-weight="bold">{html.escape(group)}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{top+panel_height}" x2="{left+panel_width}" y2="{top+panel_height}"/><line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_height}"/>')
        for replicate, (xv, yv) in enumerate(series):
            colour = colours[replicate % len(colours)]
            for xraw, yraw in zip(xv, yv):
                if xmin <= xraw <= xmax and ymin <= yraw <= ymax:
                    x = left + (xraw - xmin) / (xmax - xmin) * panel_width
                    y = top + panel_height - (yraw - ymin) / (ymax - ymin) * panel_height
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.3" fill="{colour}" opacity="0.24"/>')
        parts.append(f'<text x="{left+panel_width/2}" y="390" text-anchor="middle" font-size="11">{html.escape(xchannel)}</text>')
        parts.append(f'<text x="{left-42}" y="{top+panel_height/2}" text-anchor="middle" font-size="11" transform="rotate(-90 {left-42} {top+panel_height/2})">{html.escape(ychannel)}</text>')
    return "".join(parts), width, height


def plot_coordinates(xvalues, yvalues, left, top, width, height, limits=None):
    finite = np.isfinite(xvalues) & np.isfinite(yvalues)
    xvalues, yvalues = xvalues[finite], yvalues[finite]
    if limits is None:
        xmin, xmax = np.quantile(xvalues, [0.005, 0.995])
        ymin, ymax = np.quantile(yvalues, [0.005, 0.995])
    else:
        xmin, xmax, ymin, ymax = limits
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1
    x = left + np.clip((xvalues - xmin) / (xmax - xmin), 0, 1) * width
    y = top + height - np.clip((yvalues - ymin) / (ymax - ymin), 0, 1) * height
    return x, y, (float(xmin), float(xmax), float(ymin), float(ymax))


def axes_svg(parts, left, top, width, height, title, xlabel, ylabel):
    parts.append(f'<text x="{left+width/2}" y="{top-12}" text-anchor="middle" font-size="14" font-weight="bold">{html.escape(title)}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#FAFCFD" stroke="#718291"/>')
    parts.append(f'<text x="{left+width/2}" y="{top+height+31}" text-anchor="middle" font-size="11">{html.escape(xlabel)}</text>')
    parts.append(f'<text x="{left-38}" y="{top+height/2}" text-anchor="middle" font-size="11" transform="rotate(-90 {left-38} {top+height/2})">{html.escape(ylabel)}</text>')


def add_scatter(parts, xvalues, yvalues, left, top, width, height, colour, opacity, radius, limits=None, max_points=2500):
    count = min(max_points, len(xvalues))
    if count == 0:
        return limits
    selection = np.linspace(0, len(xvalues) - 1, count, dtype=int)
    x, y, resolved_limits = plot_coordinates(xvalues[selection], yvalues[selection], left, top, width, height, limits)
    for xpos, ypos in zip(x, y):
        parts.append(f'<circle cx="{xpos:.1f}" cy="{ypos:.1f}" r="{radius}" fill="{colour}" opacity="{opacity}"/>')
    return resolved_limits


def threshold_line(parts, value, axis, limits, left, top, width, height, colour="#A61B1B"):
    xmin, xmax, ymin, ymax = limits
    if axis == "x" and xmin <= value <= xmax:
        x = left + (value - xmin) / (xmax - xmin) * width
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+height}" stroke="{colour}" stroke-width="1.5" stroke-dasharray="5,4"/>')
    if axis == "y" and ymin <= value <= ymax:
        y = top + height - (value - ymin) / (ymax - ymin) * height
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" y2="{y:.1f}" stroke="{colour}" stroke-width="1.5" stroke-dasharray="5,4"/>')


def sample_dot_plots_svg(sample_name, dataset, thresholds, threshold_label):
    width, height = 1260, 820
    parts = [f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-weight="bold">{html.escape(sample_name)}: compensated dot plots</text>']
    parts.append(f'<text x="{width/2}" y="52" text-anchor="middle" font-size="11">Blue points are events retained by the automated QC/singlet gate; grey points are excluded or ungated context.</text>')
    fcs, gate = dataset["fcs"], dataset["gate"]
    index = {name: i for i, name in enumerate(fcs["names"])}
    data = fcs["data"]
    compensated = dataset["compensated"]
    positions = [(70, 90), (480, 90), (890, 90), (275, 465), (685, 465)]
    panel_width, panel_height = 320, 270

    specifications = [
        ("Scatter/QC gate", data[:, index[CHANNELS["fsc_a"]]], data[:, index[CHANNELS["ssc_a"]]], "FSC-A", "SSC-A", "scatter"),
        ("Singlet check", data[:, index[CHANNELS["fsc_a"]]], data[:, index[CHANNELS["fsc_h"]]], "FSC-A", "FSC-H", "singlet"),
        (f"{LABELS['af488']} versus {LABELS['af568']}", transformed(compensated[:, 1], SETTINGS["af488_cofactor"]), transformed(compensated[:, 2], SETTINGS["af568_cofactor"]), f"asinh({LABELS['af488']}/{SETTINGS['af488_cofactor']:g})", f"asinh({LABELS['af568']}/{SETTINGS['af568_cofactor']:g})", "markers"),
        (f"{LABELS['dapi']} versus {LABELS['af488']}", transformed(compensated[:, 0], SETTINGS["dapi_cofactor"]), transformed(compensated[:, 1], SETTINGS["af488_cofactor"]), f"asinh({LABELS['dapi']}/{SETTINGS['dapi_cofactor']:g})", f"asinh({LABELS['af488']}/{SETTINGS['af488_cofactor']:g})", "dapi_af488"),
        (f"{LABELS['dapi']} versus {LABELS['af568']}", transformed(compensated[:, 0], SETTINGS["dapi_cofactor"]), transformed(compensated[:, 2], SETTINGS["af568_cofactor"]), f"asinh({LABELS['dapi']}/{SETTINGS['dapi_cofactor']:g})", f"asinh({LABELS['af568']}/{SETTINGS['af568_cofactor']:g})", "dapi_af568"),
    ]
    for (left, top), (title, xvalues, yvalues, xlabel, ylabel, kind) in zip(positions, specifications):
        axes_svg(parts, left, top, panel_width, panel_height, title, xlabel, ylabel)
        if kind in ("scatter", "singlet"):
            limits = add_scatter(parts, xvalues, yvalues, left, top, panel_width, panel_height, "#A8B2BA", 0.22, 1.2, max_points=3000)
            add_scatter(parts, xvalues[gate], yvalues[gate], left, top, panel_width, panel_height, "#1769AA", 0.30, 1.25, limits=limits, max_points=2500)
        else:
            limits = add_scatter(parts, xvalues[gate], yvalues[gate], left, top, panel_width, panel_height, "#1769AA", 0.25, 1.25, max_points=3000)
            af488_threshold = float(transformed(np.array([thresholds[CHANNELS["af488"]]]), SETTINGS["af488_cofactor"])[0])
            af568_threshold = float(transformed(np.array([thresholds[CHANNELS["af568"]]]), SETTINGS["af568_cofactor"])[0])
            if kind == "markers":
                threshold_line(parts, af488_threshold, "x", limits, left, top, panel_width, panel_height)
                threshold_line(parts, af568_threshold, "y", limits, left, top, panel_width, panel_height)
            elif kind == "dapi_af488":
                threshold_line(parts, af488_threshold, "y", limits, left, top, panel_width, panel_height)
            elif kind == "dapi_af568":
                threshold_line(parts, af568_threshold, "y", limits, left, top, panel_width, panel_height)
    parts.append(f'<text x="70" y="790" font-size="11">Retained events: {int(gate.sum()):,}/{len(gate):,} ({gate.mean()*100:.1f}%). Red dashed lines: {html.escape(threshold_label)}; no DAPI gate is shown.</text>')
    return "".join(parts), width, height


def sample_histograms_svg(sample_name, dataset, pooled_background, thresholds):
    width, height = 1240, 450
    parts = [f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-weight="bold">{html.escape(sample_name)}: compensated fluorescence histograms</text>']
    positions = [55, 455, 855]
    panel_width, top, bottom = 330, 75, 360
    cofactors = [SETTINGS["dapi_cofactor"], SETTINGS["af488_cofactor"], SETTINGS["af568_cofactor"]]
    colours = ["#7A5195", "#2878B5", "#D45A3A"]
    labels = [LABELS["dapi"], LABELS["af488"], LABELS["af568"]]
    for channel_index, (left, cofactor, colour, label) in enumerate(zip(positions, cofactors, colours, labels)):
        channel = FLUORESCENCE[channel_index]
        sample_values = transformed(dataset["compensated"][dataset["gate"], channel_index], cofactor)
        reference_values = transformed(pooled_background[channel], cofactor)
        xmin = min(float(np.quantile(sample_values, 0.001)), float(np.quantile(reference_values, 0.001)))
        xmax = max(float(np.quantile(sample_values, 0.999)), float(np.quantile(reference_values, 0.999)))
        edges = np.linspace(xmin, xmax, 100)
        sample_hist = np.histogram(sample_values, bins=edges, density=True)[0]
        reference_hist = np.histogram(reference_values, bins=edges, density=True)[0]
        ymax = max(float(sample_hist.max()), float(reference_hist.max())) * 1.08
        centres = (edges[:-1] + edges[1:]) / 2
        axes_svg(parts, left, top, panel_width, bottom-top, label, f"asinh(signal/{cofactor})", "Density")
        for values, line_colour, dash in ((reference_hist, "#666666", "5,4"), (sample_hist, colour, "")):
            points = []
            for xvalue, yvalue in zip(centres, values):
                xpos = left + (xvalue - xmin) / (xmax - xmin) * panel_width
                ypos = bottom - yvalue / ymax * (bottom - top)
                points.append(f"{xpos:.1f},{ypos:.1f}")
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{line_colour}" stroke-width="2"{dash_attr}/>')
        if channel in thresholds:
            transformed_threshold = float(transformed(np.array([thresholds[channel]]), cofactor)[0])
            threshold_line(parts, transformed_threshold, "x", (xmin, xmax, 0, ymax), left, top, panel_width, bottom-top)
    parts.append('<line x1="55" y1="418" x2="82" y2="418" stroke="#666666" stroke-width="2" stroke-dasharray="5,4"/><text x="90" y="422" font-size="11">pooled matched-background reference</text>')
    parts.append('<line x1="310" y1="418" x2="337" y2="418" stroke="#1769AA" stroke-width="2"/><text x="345" y="422" font-size="11">sample distribution (panel-specific colour)</text>')
    parts.append('<line x1="620" y1="418" x2="647" y2="418" stroke="#A61B1B" stroke-width="2" stroke-dasharray="5,4"/><text x="655" y="422" font-size="11">exploratory positivity threshold; not available for DAPI</text>')
    return "".join(parts), width, height


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path, help="JSON configuration file")
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    configure(config)
    source = resolve_config_path(config_path, config["paths"]["input_dir"])
    manifest_path = resolve_config_path(config_path, config["paths"]["manifest"])
    output = resolve_config_path(config_path, config["paths"]["output_dir"])
    flowjo_targets_path = resolve_config_path(config_path, config["paths"].get("flowjo_targets"))
    if not source.is_dir():
        raise FileNotFoundError(f"Configured input directory does not exist: {source}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Configured manifest does not exist: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)

    manifest = load_manifest(manifest_path)
    print("phase: manifest loaded", flush=True)
    datasets = {}
    audit = []
    for filename, record in manifest.items():
        if record["include_for_analysis"] != "Yes":
            audit.append({"filename": filename, "status": "excluded", "reason": record["notes"]})
            continue
        path = (source / filename).resolve()
        if source != path and source not in path.parents:
            raise ValueError(f"{filename}: manifest path escapes the configured input directory")
        if not path.is_file():
            raise FileNotFoundError(f"Manifest FCS file does not exist: {path}")
        fcs = read_fcs(path)
        missing = [channel for channel in CHANNELS.values() if channel not in fcs["names"]]
        if missing:
            raise ValueError(f"{filename}: missing channels {missing}")
        gate = build_gate(fcs)
        datasets[filename] = {"fcs": fcs, "gate": gate, "manifest": record}
        audit.append({"filename": filename, "status": "read", "reason": "included by analysis manifest"})
    print(f"phase: {len(datasets)} FCS files read and gated", flush=True)

    if SETTINGS["require_matching_voltages"]:
        voltage_signatures = {
            filename: tuple(zip(dataset["fcs"]["names"], dataset["fcs"]["voltages"]))
            for filename, dataset in datasets.items()
        }
        signatures = {}
        for filename, signature in voltage_signatures.items():
            signatures.setdefault(signature, []).append(filename)
        if len(signatures) > 1:
            groups = [", ".join(files) for files in signatures.values()]
            raise ValueError("Included FCS files have mismatched detector-voltage signatures: " + " | ".join(groups))

    compensation_controls = [d for d in datasets.values() if d["manifest"]["include_for_compensation"] == "Yes"]
    backgrounds = [d for d in compensation_controls if d["manifest"]["control_group"] == CONTROL_GROUPS["background"]]
    if len(backgrounds) < 1:
        raise ValueError("At least one matched background control is required")

    def raw_channel_values(dataset, channel):
        i = dataset["fcs"]["names"].index(channel)
        return dataset["fcs"]["data"][dataset["gate"], i]

    raw_pooled_background = {
        channel: np.concatenate([raw_channel_values(dataset, channel) for dataset in backgrounds])
        for channel in FLUORESCENCE
    }
    raw_background_medians = {channel: float(np.median(values)) for channel, values in raw_pooled_background.items()}

    centroid_rows = []
    coefficient_candidates = {}
    contrasts = [
        (CHANNELS[source_key], CHANNELS[target_key], CONTROL_GROUPS[source_key])
        for source_key, target_key in COMPENSATION_PAIRS
    ]
    for source_channel, target_channel, control_group in contrasts:
        candidates = []
        controls = [d for d in compensation_controls if d["manifest"]["control_group"] == control_group]
        for number, dataset in enumerate(controls, 1):
            source_delta = float(np.median(raw_channel_values(dataset, source_channel))) - raw_background_medians[source_channel]
            target_delta = float(np.median(raw_channel_values(dataset, target_channel))) - raw_background_medians[target_channel]
            coefficient = target_delta / source_delta if source_delta > 0 else np.nan
            candidates.append(coefficient)
            centroid_rows.append({
                "source_channel": source_channel,
                "target_channel": target_channel,
                "control_replicate": number,
                "source_median_minus_background": source_delta,
                "target_median_minus_background": target_delta,
                "centroid_coefficient": coefficient,
            })
        finite = np.array([value for value in candidates if np.isfinite(value)])
        if finite.size < 1 or np.any(finite <= 0) or (finite.size > 1 and np.max(finite) / np.min(finite) > SETTINGS["maximum_replicate_ratio"]) or np.median(finite) >= SETTINGS["maximum_accepted_coefficient"]:
            raise ValueError(f"Cross-channel coefficient failed replicate QC: {source_channel} -> {target_channel}: {finite}")
        coefficient_candidates[(target_channel, source_channel)] = float(np.median(finite))

    spill_matrix = np.eye(3)
    channel_position = {channel: index for index, channel in enumerate(FLUORESCENCE)}
    for (target_channel, source_channel), coefficient in coefficient_candidates.items():
        spill_matrix[channel_position[target_channel], channel_position[source_channel]] = coefficient
    inverse_spill = np.linalg.inv(spill_matrix)
    for dataset in datasets.values():
        indices = [dataset["fcs"]["names"].index(channel) for channel in FLUORESCENCE]
        dataset["compensated"] = dataset["fcs"]["data"][:, indices] @ inverse_spill.T

    def channel_values(dataset, channel):
        return dataset["compensated"][dataset["gate"], channel_position[channel]]

    pooled_background = {
        channel: np.concatenate([channel_values(dataset, channel) for dataset in backgrounds])
        for channel in FLUORESCENCE
    }
    thresholds = {
        CHANNELS["af488"]: float(np.quantile(pooled_background[CHANNELS["af488"]], SETTINGS["background_percentile"])),
        CHANNELS["af568"]: float(np.quantile(pooled_background[CHANNELS["af568"]], SETTINGS["background_percentile"])),
    }
    background_thresholds = dict(thresholds)
    calibration_rows = []
    threshold_mode = f"pooled background {SETTINGS['background_percentile'] * 100:g}th percentile"
    threshold_config_mode = config.get("thresholds", {}).get("mode", "background_percentile")
    if threshold_config_mode == "flowjo_calibrated":
        if not flowjo_targets_path.is_file():
            raise FileNotFoundError(f"Configured FlowJo target file does not exist: {flowjo_targets_path}")
        targets = load_flowjo_targets(flowjo_targets_path)
        thresholds, calibration_rows = calibrate_quadrant_thresholds(datasets, targets, channel_position)
        threshold_mode = "shared FlowJo-calibrated quadrant thresholds"
    background_medians = {channel: float(np.median(values)) for channel, values in pooled_background.items()}
    print("phase: background thresholds calculated", flush=True)

    stats = []
    for filename, dataset in datasets.items():
        role = dataset["manifest"]["role"]
        row = {
            "filename": filename,
            "role": role,
            "control_group": dataset["manifest"]["control_group"],
            "events_total": dataset["fcs"]["data"].shape[0],
            "events_gated": int(dataset["gate"].sum()),
            "events_gated_pct": float(dataset["gate"].mean() * 100),
        }
        for label, channel in (("dapi", CHANNELS["dapi"]), ("af488", CHANNELS["af488"]), ("af568", CHANNELS["af568"])):
            values = channel_values(dataset, channel)
            summary = percentile_summary(values)
            for metric, value in summary.items():
                row[f"{label}_{metric}"] = value
            channel_index = dataset["fcs"]["names"].index(channel)
            channel_maximum = dataset["fcs"]["ranges"][channel_index] - 1
            row[f"{label}_saturated_pct"] = float(np.mean(values >= channel_maximum) * 100)
            if label in ("af488", "af568"):
                row[f"{label}_net_median"] = summary["median"] - background_medians[channel]
                row[f"{label}_fold_background"] = (
                    summary["median"] / background_medians[channel]
                    if background_medians[channel] != 0 else np.nan
                )
                row[f"{label}_positive_pct"] = float(np.mean(values > thresholds[channel]) * 100)
        af488_values = channel_values(dataset, CHANNELS["af488"])
        af568_values = channel_values(dataset, CHANNELS["af568"])
        af488_positive = af488_values > thresholds[CHANNELS["af488"]]
        af568_positive = af568_values > thresholds[CHANNELS["af568"]]
        row["quadrant_double_negative_pct"] = float(np.mean(~af488_positive & ~af568_positive) * 100)
        row["quadrant_marker_2_only_pct"] = float(np.mean(~af488_positive & af568_positive) * 100)
        row["quadrant_marker_1_only_pct"] = float(np.mean(af488_positive & ~af568_positive) * 100)
        row["quadrant_double_positive_pct"] = float(np.mean(af488_positive & af568_positive) * 100)
        stats.append(row)
    print("phase: sample statistics calculated", flush=True)

    compensation_rows = []
    control_sources = {
        CHANNELS["dapi"]: CONTROL_GROUPS["dapi"],
        CHANNELS["af488"]: CONTROL_GROUPS["af488"],
        CHANNELS["af568"]: CONTROL_GROUPS["af568"],
    }
    seed = 20260611
    for source_channel, control_group in control_sources.items():
        controls = [d for d in compensation_controls if d["manifest"]["control_group"] == control_group]
        for target_channel in FLUORESCENCE:
            if target_channel == source_channel:
                continue
            replicate_estimates = []
            for number, dataset in enumerate(controls):
                source_values = raw_channel_values(dataset, source_channel)
                target_values = raw_channel_values(dataset, target_channel)
                slope, lower, upper, correlation = slope_with_bootstrap(source_values, target_values, seed + number)
                replicate_estimates.append(slope)
                applied = coefficient_candidates.get((target_channel, source_channel), 0)
                compensation_rows.append({
                    "source_channel": source_channel,
                    "target_channel": target_channel,
                    "control_file": dataset["manifest"]["replicate_group"] + f" replicate {number + 1}",
                    "diagnostic_slope": slope,
                    "ci95_lower": lower,
                    "ci95_upper": upper,
                    "pearson_correlation": correlation,
                    "applied_coefficient": applied,
                    "decision": "Applied from replicate centroid contrast" if applied else "Not applied: no matched negative contrast; within-cell slope is diagnostic only",
                })
    print("phase: compensation diagnostics calculated", flush=True)

    matrix_rows = [{"detector": FLUORESCENCE[i], **{FLUORESCENCE[j]: float(spill_matrix[i, j]) for j in range(3)}} for i in range(3)]
    matrix_fields = ["detector"] + FLUORESCENCE
    write_csv(output / "compensation_matrix_provisional.csv", matrix_rows, matrix_fields)
    write_csv(output / "compensation_centroid_estimates.csv", centroid_rows, list(centroid_rows[0].keys()))
    write_csv(output / "compensation_diagnostic_slopes.csv", compensation_rows, list(compensation_rows[0].keys()))
    write_csv(output / "sample_statistics.csv", stats, list(stats[0].keys()))
    write_csv(output / "read_audit.csv", audit, ["filename", "status", "reason"])
    if calibration_rows:
        write_csv(output / "flowjo_calibrated_comparison.csv", calibration_rows, list(calibration_rows[0].keys()))
    write_csv(output / "thresholds.csv", [{
        "threshold_mode": threshold_mode,
        "marker_1_threshold": thresholds[CHANNELS["af488"]],
        "marker_2_threshold": thresholds[CHANNELS["af568"]],
        "background_marker_1_reference": background_thresholds[CHANNELS["af488"]],
        "background_marker_2_reference": background_thresholds[CHANNELS["af568"]],
    }], ["threshold_mode", "marker_1_threshold", "marker_2_threshold", "background_marker_1_reference", "background_marker_2_reference"])
    print("phase: CSV outputs written", flush=True)

    sample_rows = [row for row in stats if row["role"] == "Sample"]
    labels = [Path(row["filename"]).stem for row in sample_rows]
    x = np.arange(len(labels))
    body, width, height = bar_panel_svg(sample_rows, labels)
    save_svg(plot_dir / "sample_marker_summary.svg", body, width, height)
    print("phase: sample plot written", flush=True)

    body, width, height = distribution_svg(datasets, channel_values)
    save_svg(plot_dir / "control_distributions.svg", body, width, height)
    print("phase: distribution plot written", flush=True)

    body, width, height = scatter_svg(datasets, channel_values)
    save_svg(plot_dir / "control_cross_channel_diagnostics.svg", body, width, height)
    print("phase: cross-channel plot written", flush=True)

    sample_plot_dir = output / "sample_plots"
    sample_plot_dir.mkdir(exist_ok=True)
    sample_links = []
    used_slugs = set()
    for row in sample_rows:
        filename = row["filename"]
        dataset = datasets[filename]
        sample_name = Path(filename).stem
        slug = "".join(character if character.isalnum() or character in "-_" else "_" for character in sample_name)
        if slug in used_slugs:
            raise ValueError(f"Sample filenames produce a duplicate plot name: {sample_name}")
        used_slugs.add(slug)
        dot_name = f"{slug}_dot_plots.svg"
        histogram_name = f"{slug}_histograms.svg"
        page_name = f"{slug}.html"
        body, width, height = sample_dot_plots_svg(sample_name, dataset, thresholds, threshold_mode)
        save_svg(sample_plot_dir / dot_name, body, width, height)
        body, width, height = sample_histograms_svg(sample_name, dataset, pooled_background, thresholds)
        save_svg(sample_plot_dir / histogram_name, body, width, height)
        sample_page = f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(sample_name)} flow cytometry plots</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;max-width:1300px;margin:24px auto;color:#17202a}}h1{{color:#17365d}}img{{width:100%;border:1px solid #ccd6df;margin-bottom:22px}}.note{{background:#fff2cc;padding:10px;border-left:4px solid #c69214}}</style></head><body>
<h1>{html.escape(sample_name)}</h1><p class="note">Provisional compensation: {html.escape(LABELS['af488'])} to {html.escape(CHANNELS['af568'])} = {spill_matrix[2,1]*100:.3f}%; {html.escape(LABELS['af568'])} to {html.escape(CHANNELS['af488'])} = {spill_matrix[1,2]*100:.3f}%. {html.escape(LABELS['dapi'])} spillover is zero unless supported by configured matched controls. Gate mode: {html.escape(threshold_mode)}.</p>
<img src="{dot_name}" alt="{html.escape(sample_name)} dot plots"><img src="{histogram_name}" alt="{html.escape(sample_name)} histograms"></body></html>'''
        (sample_plot_dir / page_name).write_text(sample_page, encoding="utf-8")
        sample_links.append((sample_name, page_name))
    sample_index = f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(ANALYSIS_TITLE)} sample plots</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;max-width:900px;margin:32px auto;color:#17202a}}h1{{color:#17365d}}li{{margin:12px 0}}a{{color:#1769aa;font-size:18px}}</style></head><body><h1>{html.escape(ANALYSIS_TITLE)}: sample dot plots and histograms</h1><ul>''' + "".join(
        f'<li><a href="{html.escape(page)}">{html.escape(name)}</a></li>' for name, page in sample_links
    ) + "</ul></body></html>"
    (sample_plot_dir / "index.html").write_text(sample_index, encoding="utf-8")
    print(f"phase: {len(sample_links)} sample plot sets written", flush=True)

    warnings = []
    for row in stats:
        if row["dapi_saturated_pct"] > 5:
            warnings.append(f"{row['filename']}: DAPI saturation {row['dapi_saturated_pct']:.2f}%")
        if row["events_gated_pct"] < 70:
            warnings.append(f"{row['filename']}: only {row['events_gated_pct']:.1f}% retained by automated QC gate")

    report = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(ANALYSIS_TITLE)}</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Arial,sans-serif;max-width:1100px;margin:32px auto;color:#17202a;line-height:1.45}}h1,h2{{color:#17365d}}.warning{{background:#fff2cc;padding:12px;border-left:4px solid #c69214}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ccd6df;padding:6px;text-align:left}}th{{background:#17365d;color:white}}img{{max-width:100%;border:1px solid #ccd6df}}</style></head><body>
<h1>{html.escape(ANALYSIS_TITLE)}</h1>
<p><strong>Dataset:</strong> {len(datasets)} included files; raw FCS files read-only.</p>
<div class="warning"><strong>Compensation decision:</strong> provisional cross-channel coefficients were estimated by contrasting each configured positive-control centroid with the pooled matched-background centroid and taking the median across technical replicates. {html.escape(LABELS['af488'])} to {html.escape(CHANNELS['af568'])} = {spill_matrix[2,1]:.5f}; {html.escape(LABELS['af568'])} to {html.escape(CHANNELS['af488'])} = {spill_matrix[1,2]:.5f}. Unconfigured coefficients remain zero. Within-cell regression slopes are reported only as diagnostics because they include biological and autofluorescence covariance.</div>
<h2>Positivity thresholds</h2>
<p>{html.escape(LABELS['af488'])} positive threshold = {thresholds[CHANNELS['af488']]:.3f}; {html.escape(LABELS['af568'])} positive threshold = {thresholds[CHANNELS['af568']]:.3f}. Threshold mode: {html.escape(threshold_mode)}. Configured pooled-background reference values were {html.escape(LABELS['af488'])} = {background_thresholds[CHANNELS['af488']]:.3f} and {html.escape(LABELS['af568'])} = {background_thresholds[CHANNELS['af568']]:.3f}.</p>
<h2>Sample marker summary</h2><img src="plots/sample_marker_summary.svg" alt="Sample marker summary">
<h2>Per-sample plots</h2><p><a href="sample_plots/index.html">Open dot plots and histograms for each biological sample</a>.</p>
<h2>Control distributions</h2><img src="plots/control_distributions.svg" alt="Control distributions">
<h2>Cross-channel diagnostics</h2><img src="plots/control_cross_channel_diagnostics.svg" alt="Cross-channel diagnostics">
<h2>QC warnings</h2><ul>{''.join(f'<li>{html.escape(w)}</li>' for w in warnings) if warnings else '<li>None at configured thresholds.</li>'}</ul>
<h2>Interpretation boundaries</h2><ul><li>The automated gate removes time endpoints, non-finite/non-positive scatter events, robust scatter outliers and FSC-A/FSC-H singlet-ratio outliers.</li><li>No {html.escape(LABELS['dapi'])}-positive gate is calculated by this release.</li><li>Positive percentages are exploratory and require biological validation. FlowJo-calibrated thresholds are fitted to the supplied manual reference and are not independent validation.</li><li>No raw event table or compensated FCS file was exported.</li></ul>
</body></html>"""
    (output / "report.html").write_text(report, encoding="utf-8")

    run = {
        "source": str(source),
        "manifest": str(manifest_path),
        "files_included": len(datasets),
        "files_excluded": len([x for x in audit if x["status"] == "excluded"]),
        "compensation_matrix": "replicate centroid-derived (provisional)",
        "marker_1_to_marker_2": float(spill_matrix[2, 1]),
        "marker_2_to_marker_1": float(spill_matrix[1, 2]),
        "background_threshold_percentile": SETTINGS["background_percentile"] * 100,
        "threshold_mode": threshold_mode,
        "marker_1_threshold": thresholds[CHANNELS["af488"]],
        "marker_2_threshold": thresholds[CHANNELS["af568"]],
        "raw_events_exported": False,
        "sample_plot_sets": len(sample_links),
    }
    (output / "run_summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
