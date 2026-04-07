# CAST-ECG: Non-Contact ECG Reconstruction from Radar Signals

A physics-aware, morphology-constrained deep learning model for reconstructing ECG waveforms and hemodynamic signals from contactless radar data.

---

## Overview

CAST-ECG takes 4-channel multiband radar input (phase & magnitude for cardiac and respiratory bands) and simultaneously reconstructs:

- **ECG** — Leads I and II
- **Blood Pressure (BP)** waveform
- **ICG** (Impedance Cardiography) waveform
- **Cardiac Flow** (dICG/dt)
- **R-peak locations**

The model is evaluated across three physiological maneuver conditions: **Resting**, **Valsalva**, and **Apnea**.

---

## Model Architecture

CAST-ECG is built around a dual-branch encoder with a U-Net decoder.

```
Radar Input (4ch, 128 Hz)
        │
    [Input Conv]
        │
  ┌─────┴──────┐
  │            │
[Dilated     [Learned
 Inception]   Filterbank]
 (time domain) (freq domain)
  │            │
  └──[SpatioTemporal Router]──┘
           │
     [U-Net Decoder]
      (4 scales, skip connections)
           │
   ┌───────┼────────┐
  ECG     BP    ICG / Flow
(+ Phase Shift  (+ Peak Head)
 Compensator)
```

**Key components:**
- **DilatedInceptionBranch** — multi-scale time-domain feature extraction
- **LearnedFilterBank** — spectral feature extraction with learnable filters
- **SpatioTemporalRouter** — context-aware dynamic gating between branches (also used for XAI)
- **LearnablePhaseShift** — FFT-based alignment layer for ECG output

---

## Repository Structure

```
ECG-RECONSTRUCTION/
├── models/
│   ├── cast_ecg.py           # Main model (SimplifiedCASTECG_Paper)
│   ├── incept.py             # Dilated Inception branch
│   ├── filterbank_branch.py  # Learned Filterbank branch
│   └── router.py             # SpatioTemporal Router
│
├── configs/
│   └── config.py             # Hyperparameters and paths
│
├── dataload/
│   ├── dataset.py            # Patient-wise train/val/test splits
│   └── dataset_loso.py       # Leave-One-Subject-Out splits
│
├── utils/
│   ├── losses.py             # CompleteLoss (multi-task weighted loss)
│   ├── metrics.py            # PCC, temporal/spectral correlation, RMSE, MAE
│   ├── xai.py                # Explainability via router gate maps
│   ├── visualize.py          # Waveform and result plotting
│   └── radar_preprocessing.py
│
├── train.py                  # Training script
├── complete_test.py          # Full evaluation pipeline
├── final_test.py             # Final inference script
├── master_test_with_xai.py   # Testing with XAI gate visualizations
├── comprehensive_evaluation.py
├── ablation_window_size.py   # Window size ablation study
├── preprocess.py
├── postprocess.py
│
├── weights_resting.txt       # Optimized loss weights — Resting
├── weights_apnea.txt         # Optimized loss weights — Apnea
├── weights_valsalva.txt      # Optimized loss weights — Valsalva
│
├── Resting test res/         # Test outputs — Resting condition
├── Apnea test Res/           # Test outputs — Apnea condition
├── Valsalva test res/        # Test outputs — Valsalva condition
└── individual_ecgres/        # Per-subject ECG analysis
```

---

## Data Format

Input: `.h5` files with 4-channel radar signals at 128 Hz.

```
Filename pattern: multiband_4ch_128hz_{maneuver}.h5
Channels:
  0 — Phase (Heart band)
  1 — Magnitude (Heart band)
  2 — Phase (Resp band)
  3 — Magnitude (Resp band)

Window size: 1310 samples (~10.2 s)
Stride:      655 samples (50% overlap)
```

Place data files under `../../data/` relative to the repo root (configurable in `configs/config.py`).

---

## Training

```bash
python train.py
```

Key config options in `configs/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `maneuvers_to_load` | `[('1', 'Resting')]` | Which condition(s) to train on |
| `epochs` | 100 | Training epochs |
| `batch_size` | 16 | Batch size |
| `lr` | 1e-3 | Learning rate |
| `checkpoint_dir` | `checkpoints_multiband` | Where to save model weights |

Condition-specific optimized hyperparameters are documented in `weights_resting.txt`, `weights_apnea.txt`, and `weights_valsalva.txt`.

---

## Evaluation

```bash
# Standard test + per-lead metrics
python complete_test.py

# With XAI gate map visualizations
python master_test_with_xai.py

# Comprehensive per-subject analysis
python comprehensive_evaluation.py
```

---

## Results

Test results for each condition are organized as follows:

```
{Condition} test res/
├── standard_test_evaluation/   # Sample plots + per-lead CSV metrics
├── master_xai_results/         # XAI gate map overlays
├── forensic_best_window/       # Best reconstructed window visualization
└── forensic_results_pdf/       # PDF reports of best windows
```

Individual subject-level analysis (ECG waveforms, peak detection, noise masks) is in `individual_ecgres/`.

---

## Loss Function

`CompleteLoss` is a weighted multi-task loss combining:

| Component | Target |
|---|---|
| L1 | Signal reconstruction |
| Peak loss | R-peak localization |
| Temporal correlation | Time-domain alignment |
| Spectral correlation | Frequency-domain fidelity |
| Slope loss | Morphological sharpness |
| Total variation | Signal smoothness |

---

## Metrics

| Metric | Description |
|---|---|
| ECG Temporal Corr | Pearson correlation of time-aligned signals |
| ECG Spectral Corr | Cosine similarity of normalized FFT magnitudes |
| ECG PCC | Pearson correlation coefficient |
| ECG RMSE | Root mean square error |
| BP SBP/DBP MAE | Systolic / diastolic BP error |
| ICG / Flow PCC | Correlation for hemodynamic signals |
