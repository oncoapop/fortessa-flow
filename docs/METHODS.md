# Methods and assumptions

## Input

The workflow reads list-mode FCS 3.0 files with floating-point, double-precision or fixed-width integer DATA segments. The source files are opened read-only. Event-level tables and compensated FCS files are not exported.

By default, all included files must have identical parameter-name and detector-voltage signatures. Files marked `include_for_compensation=No` are never used to estimate compensation or matched-background thresholds.

## Automated QC gate

The default gate:

1. removes events with non-finite or non-positive FSC/SSC values;
2. trims the configured fraction from each acquisition-time endpoint;
3. removes robust FSC-A and SSC-A outliers using median absolute deviation;
4. removes FSC-A/FSC-H ratio outliers as a conservative singlet check.

This is not a replacement for assay-specific manual gating.

## Compensation

For each configured source-to-target pair, the workflow calculates:

`coefficient = (target positive-control median - target background median) / (source positive-control median - source background median)`

The median coefficient across technical replicates is applied. Replicate disagreement and implausibly large coefficients stop the run. Within-cell regression slopes are exported only as diagnostics because biological signal and autofluorescence can covary within cell-based controls.

## Positivity thresholds

The independent default is a configured percentile of pooled matched-background controls. An optional FlowJo calibration mode fits one shared vertical and horizontal gate to supplied manual quadrant percentages. Calibrated outputs must be described as fitted to the manual reference and must not be presented as independent validation.

## Limitations

- The current release assumes one DAPI-like channel and two marker channels.
- Cell-based omitted-primary controls are less clean than matched single-colour compensation beads.
- Compensation cannot be rescued reliably when controls use different detector voltages.
- A missing matched negative control leaves the corresponding spillover coefficients unsupported.
- Automated thresholds and gates require inspection before biological interpretation, publication or clinical use.
- This software is for research use only and is not a medical device.
