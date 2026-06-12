# fortessa-flow

`fortessa-flow` is a local, configuration-driven workflow for FCS 3.0 quality control, provisional three-colour compensation, conservative gating, quadrant statistics, dot plots and fluorescence histograms.

It was designed for inspectable research workflows where raw FCS files must remain unchanged. It is **research software**, not a substitute for manual review and not for clinical decision-making.

## Features

- Read-only FCS 3.0 parser.
- Configurable paths, channel names, labels, control groups and gate parameters.
- Replicate-aware centroid compensation with fail-fast QC.
- Time, scatter and FSC-A/FSC-H singlet QC.
- Background-percentile or optional FlowJo-calibrated shared quadrant gates.
- Per-sample dot plots, histograms, compensation diagnostics and CSV statistics.
- No raw event-table or compensated-FCS export.
- Synthetic tests; no real cytometry data is distributed.

## Requirements

- Python 3.10 or later.
- NumPy 1.24 or later.
- macOS, Linux or another environment supported by Python and NumPy.

## Installation

```zsh
git clone https://github.com/oncoapop/fortessa-flow.git
cd fortessa-flow
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

No Homebrew or administrator access is required.

## Configuration

Copy the examples without editing the originals:

```zsh
cp configs/example_config.json local_config.json
cp configs/example_manifest.csv manifest.csv
```

Set all local paths in `local_config.json`:

```json
{
  "paths": {
    "input_dir": "/path/to/fcs/files",
    "manifest": "/path/to/manifest.csv",
    "output_dir": "/path/to/results",
    "flowjo_targets": null
  }
}
```

The complete example also defines detector/channel names, display labels, control-group names, compensation pairs, voltage-matching QC, gate limits, positivity percentiles and transformation cofactors. Relative paths are resolved relative to the configuration file.

The manifest uses one row per FCS file. Control-group strings must exactly match those in the configuration. Use coded sample IDs and avoid confidential identifiers.

## Run

```zsh
fortessa-flow --config local_config.json
```

The output directory contains:

- `report.html` and `sample_plots/index.html`;
- compensated aggregate `sample_statistics.csv`;
- provisional compensation matrix and replicate estimates;
- non-applied within-cell diagnostic slopes;
- thresholds and read audit metadata;
- SVG dot plots and histograms for each sample.

## Optional FlowJo calibration

Set `thresholds.mode` to `flowjo_calibrated` and `paths.flowjo_targets` to a CSV modelled on `configs/example_flowjo_targets.csv`. The optimiser fits one shared marker-1 and marker-2 threshold across all reference samples. It does not tune each sample separately.

FlowJo-calibrated output is fitted to the supplied manual percentages and must be labelled accordingly. Keep an independent background-derived analysis for comparison.

## Validation and interpretation

Before using results:

1. Verify controls and samples used identical detector settings.
2. Inspect compensation replicate agreement and omitted-channel residuals.
3. Review every scatter, singlet and fluorescence plot.
4. Confirm biological gates manually on representative samples.
5. Report whether gates were background-derived or FlowJo-calibrated.

See [Methods and assumptions](docs/METHODS.md) for formulas and limitations.

## Test

```zsh
python3 -m pip install -e '.[test]'
pytest -q
```

Tests generate synthetic FCS files in a temporary directory. They do not require or download real data.

## Data security

FCS metadata and filenames can contain confidential identifiers. The `.gitignore` excludes FCS files, FlowJo workspaces, local configuration and generated outputs. Review `git status` before every commit. See [SECURITY.md](SECURITY.md).

## Licence

MIT. See [LICENSE](LICENSE).
