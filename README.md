# Fan & Activity Anomaly Detection — Arduino Nano 33 BLE Sense

**EE 446: Tiny Machine Learning for Ultra Low-Power Edge Computing | University of Washington, Spring 2026**

Two anomaly-detection pipelines on the same board: a mechanical fan-monitoring system built in Edge Impulse (spectral analysis + Keras classifier + k-means anomaly scoring), and a human-activity autoencoder that flags unusual motion patterns from IMU data.

---

## Part 1 — Fan monitoring (Edge Impulse)

Detects abnormal fan operation from onboard IMU vibration signatures. Built from Edge Impulse's public [Fan Monitoring example project](https://studio.edgeimpulse.com/public/47996/latest), adapted for time-series vibration data:

1. **Time-series data** — vibration recordings under normal and faulty fan conditions
2. **Spectral analysis** — frequency-domain feature extraction from the raw vibration signal
3. **Classification (Keras)** — a neural network trained to distinguish fan states
4. **Anomaly detection (k-means)** — clustering on the feature space flags operating conditions that don't match any known-good cluster, catching failure modes not seen during training

Two firmware builds are included (`v10`, `v11`). These are same-day rebuilds of the same Edge Impulse export — the compiled binaries are functionally identical (only the embedded build timestamp differs), not separate model iterations. Either works; `v11` is kept as the more recent build.

---

## Part 2 — Human activity anomaly detection (local notebook)

Trains an autoencoder on the [mHealth dataset](https://doi.org/10.1109/BSN.2014.24) (Subject 6) to reconstruct "normal" activity IMU windows; high reconstruction error at inference time flags anomalous motion.

The mHealth dataset (`data/mHealth_subject6.log`) contains 50 Hz recordings from sensors on the chest, right wrist, and left ankle across 12 activities (standing, walking, running, cycling, etc. — full label list in `Dataset-README.txt`). This is the same dataset used in the companion [ensemble-activity-classifier](https://github.com/sdadhich04/ensemble-activity-classifier) repo (Lab 8), which does supervised classification on the same data rather than anomaly detection.

The training data's acceleration columns are in m/s², while the Arduino's IMU library reports acceleration in g's — `Activity_Anomaly.ino` converts on read (×9.80665) so live inference sees the same units the model was calibrated on. The anomaly threshold (`kReconstructionErrorThreshold`) is set to mean + 2·std of the training set's reconstruction error; see the notebook's threshold-selection cell if you retrain and need to recompute it.

---

## Repository contents

```
TinyML_Lab6_Student_TODO.ipynb        ← Notebook: load mHealth data → train autoencoder → export fp32/int8 TFLite
TinyML-Lab#6.pdf                      ← Lab instructions (both parts)
EE446_Lab6_Submission_Guidelines.pdf  ← Submission rubric
Lab6_EE446_Report.pdf                 ← Written report
Dataset-README.txt                    ← mHealth dataset documentation (columns, activity labels, sensor placement)
data/
  mHealth_subject6.log                ← Raw IMU + ECG log, Subject 6 (18 MB)
models/
  autoencoder_fp32.tflite             ← Float32 autoencoder
  autoencoder_int8.tflite             ← Int8 quantized autoencoder
arduino/
  activity_anomaly/
    Activity_Anomaly.ino              ← Inference sketch — loads autoencoder_model.cc, flags high reconstruction error
    autoencoder_model.cc              ← Autoencoder as C array (must stay alongside the .ino to compile)
fan_monitoring/
  v10/                                ← Edge Impulse firmware build v10 (.bin + flash scripts)
  v11/                                ← Edge Impulse firmware build v11 (.bin + flash scripts)
```

---

## Quick start

### Part 1: Flash fan monitoring firmware

```bash
# Windows
fan_monitoring\v11\flash_windows.bat

# Mac
fan_monitoring/v11/flash_mac.command

# Linux
bash fan_monitoring/v11/flash_linux.sh
```

Both builds are functionally identical (see note above) — `v11/` is used below for convenience.

### Part 2: Run the notebook (train the autoencoder)

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
jupyter notebook TinyML_Lab6_Student_TODO.ipynb
```

Expects `mHealth_subject6.log` in the same folder as the notebook — copy it in from `data/` or point the notebook's path there.

### Part 2: Flash the activity anomaly sketch

1. Install the `TensorFlowLite` and `Arduino_BMI270_BMM150` (or `Arduino_LSM9DS1`) libraries
2. Open `arduino/activity_anomaly/Activity_Anomaly.ino` in Arduino IDE — `autoencoder_model.cc` is already alongside it
3. Upload to Nano 33 BLE Sense, open Serial Monitor to see reconstruction-error-based anomaly flags

---

## Hardware

- **Arduino Nano 33 BLE Sense** (Nordic nRF52840, 1 MB flash, 256 KB RAM, onboard IMU)

---

## Authors

Sparsh Dadhich — University of Washington, ECE / Neuroscience

---

## License

MIT — see [LICENSE](LICENSE). This covers the author's own code, notebooks, and documentation in this repo.
