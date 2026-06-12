import csv
import json
from pathlib import Path

import numpy as np

from fortessa_flow.cli import main, read_fcs


CHANNELS = ["FSC-A", "FSC-H", "SSC-A", "V 450/50-A", "Blue 530/30-A", "YG 586/15-A", "Time"]


def write_fcs(path, data, voltage="300"):
    text_start = 58
    parameter_count = data.shape[1]

    def make_text(data_start, data_end):
        pairs = [
            ("$BEGINANALYSIS", "0"), ("$ENDANALYSIS", "0"),
            ("$BEGINDATA", f"{data_start:08d}"), ("$ENDDATA", f"{data_end:08d}"),
            ("$BYTEORD", "1,2,3,4"), ("$DATATYPE", "F"),
            ("$MODE", "L"), ("$NEXTDATA", "0"),
            ("$PAR", str(parameter_count)), ("$TOT", str(data.shape[0])),
        ]
        for index, channel in enumerate(CHANNELS, 1):
            pairs.extend([
                (f"$P{index}B", "32"), (f"$P{index}E", "0,0"),
                (f"$P{index}N", channel), (f"$P{index}R", "262144"),
                (f"$P{index}V", str(voltage)),
            ])
        return ("|" + "|".join(item for pair in pairs for item in pair) + "|").encode("latin-1")

    text = make_text(0, 0)
    data_start = text_start + len(text)
    raw_data = np.asarray(data, dtype="<f4").tobytes()
    data_end = data_start + len(raw_data) - 1
    text = make_text(data_start, data_end)
    data_start = text_start + len(text)
    data_end = data_start + len(raw_data) - 1
    text = make_text(data_start, data_end)
    text_end = text_start + len(text) - 1
    header = (
        b"FCS3.0" + b" " * 4
        + f"{text_start:>8}".encode()
        + f"{text_end:>8}".encode()
        + f"{data_start:>8}".encode()
        + f"{data_end:>8}".encode()
        + b"       0" + b"       0"
    )
    path.write_bytes(header + text + raw_data)


def events(seed, marker_1=100, marker_2=120, count=1200):
    rng = np.random.default_rng(seed)
    fsc = np.clip(rng.normal(80000, 12000, count), 1000, 250000)
    fsc_h = np.clip(fsc * rng.normal(0.92, 0.025, count), 1000, 250000)
    ssc = np.clip(rng.normal(35000, 9000, count), 1000, 250000)
    dapi = np.clip(rng.normal(70000, 15000, count), 0, 250000)
    m1 = np.clip(rng.normal(marker_1, max(marker_1 * 0.18, 20), count), 0, 250000)
    m2 = np.clip(rng.normal(marker_2, max(marker_2 * 0.18, 20), count), 0, 250000)
    time = np.linspace(0, 1000, count)
    return np.column_stack([fsc, fsc_h, ssc, dapi, m1, m2, time])


def test_read_and_run_workflow(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    specifications = {
        "background_1.fcs": (100, 120),
        "background_2.fcs": (105, 115),
        "marker_1_1.fcs": (2500, 120 + 0.05 * (2500 - 100)),
        "marker_1_2.fcs": (2400, 115 + 0.05 * (2400 - 100)),
        "marker_2_1.fcs": (100 + 0.012 * (3000 - 120), 3000),
        "marker_2_2.fcs": (105 + 0.012 * (2900 - 115), 2900),
        "excluded_bad_control.fcs": (9000, 9000),
        "sample_01.fcs": (600, 1000),
    }
    for seed, (filename, values) in enumerate(specifications.items(), 1):
        write_fcs(data_dir / filename, events(seed, *values))

    parsed = read_fcs(data_dir / "sample_01.fcs")
    assert parsed["data"].shape == (1200, 7)
    assert parsed["names"] == CHANNELS

    manifest = tmp_path / "manifest.csv"
    rows = [
        ("background_1.fcs", "Control", "Background", "background"),
        ("background_2.fcs", "Control", "Background", "background"),
        ("marker_1_1.fcs", "Control", "Marker 1 control", "marker_1"),
        ("marker_1_2.fcs", "Control", "Marker 1 control", "marker_1"),
        ("marker_2_1.fcs", "Control", "Marker 2 control", "marker_2"),
        ("marker_2_2.fcs", "Control", "Marker 2 control", "marker_2"),
        ("excluded_bad_control.fcs", "Control", "Marker 1 control", "excluded"),
        ("sample_01.fcs", "Sample", "", ""),
    ]
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "role", "control_group", "replicate_group", "include_for_compensation", "include_for_analysis", "notes"])
        for row in rows:
            include_compensation = "Yes" if row[1] == "Control" and row[0] != "excluded_bad_control.fcs" else "No"
            writer.writerow([*row, include_compensation, "Yes", "synthetic"])

    output = tmp_path / "results"
    config = {
        "analysis_title": "Synthetic test",
        "paths": {"input_dir": "data", "manifest": "manifest.csv", "output_dir": "results", "flowjo_targets": None},
        "channels": {"fsc_a": "FSC-A", "fsc_h": "FSC-H", "ssc_a": "SSC-A", "dapi": "V 450/50-A", "marker_1": "Blue 530/30-A", "marker_2": "YG 586/15-A", "time": "Time"},
        "labels": {"dapi": "DAPI", "marker_1": "Marker 1", "marker_2": "Marker 2"},
        "control_groups": {"background": "Background", "dapi": "DAPI-only", "marker_1": "Marker 1 control", "marker_2": "Marker 2 control"},
        "compensation": {"pairs": [{"source": "marker_1", "target": "marker_2"}, {"source": "marker_2", "target": "marker_1"}], "maximum_accepted_coefficient": 0.2, "maximum_replicate_ratio": 2.0},
        "gating": {"time_trim_fraction": 0.005, "scatter_mad_limit": 4.0, "singlet_mad_limit": 3.5},
        "qc": {"require_matching_voltages": True},
        "thresholds": {"mode": "background_percentile", "background_percentile": 0.99},
        "transforms": {"dapi_cofactor": 1000, "marker_1_cofactor": 100, "marker_2_cofactor": 100},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    main(["--config", str(config_path)])

    assert (output / "report.html").exists()
    assert (output / "sample_plots" / "sample_01.html").exists()
    assert (output / "thresholds.csv").exists()
    matrix = list(csv.DictReader((output / "compensation_matrix_provisional.csv").open()))
    marker_1_to_2 = float(matrix[2]["Blue 530/30-A"])
    marker_2_to_1 = float(matrix[1]["YG 586/15-A"])
    assert 0.03 < marker_1_to_2 < 0.08
    assert 0.005 < marker_2_to_1 < 0.03


def test_invalid_threshold_mode_fails(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "paths": {"input_dir": "data", "manifest": "manifest.csv", "output_dir": "results"},
        "channels": {},
        "control_groups": {},
        "thresholds": {"mode": "unsupported"},
    }))
    try:
        main(["--config", str(config_path)])
    except ValueError as exc:
        assert "Unsupported threshold mode" in str(exc)
    else:
        raise AssertionError("Invalid threshold mode should fail")


def test_mismatched_voltages_fail(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_fcs(data_dir / "sample_a.fcs", events(1), voltage="300")
    write_fcs(data_dir / "sample_b.fcs", events(2), voltage="301")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "role", "control_group", "replicate_group", "include_for_compensation", "include_for_analysis", "notes"])
        writer.writerow(["sample_a.fcs", "Sample", "", "", "No", "Yes", "synthetic"])
        writer.writerow(["sample_b.fcs", "Sample", "", "", "No", "Yes", "synthetic"])
    config = {
        "paths": {"input_dir": "data", "manifest": "manifest.csv", "output_dir": "results"},
        "channels": {"fsc_a": "FSC-A", "fsc_h": "FSC-H", "ssc_a": "SSC-A", "dapi": "V 450/50-A", "marker_1": "Blue 530/30-A", "marker_2": "YG 586/15-A", "time": "Time"},
        "control_groups": {"background": "Background", "dapi": "DAPI-only", "marker_1": "Marker 1 control", "marker_2": "Marker 2 control"},
        "compensation": {"pairs": [{"source": "marker_1", "target": "marker_2"}]},
        "qc": {"require_matching_voltages": True},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    try:
        main(["--config", str(config_path)])
    except ValueError as exc:
        assert "mismatched detector-voltage signatures" in str(exc)
    else:
        raise AssertionError("Mismatched voltages should fail")
