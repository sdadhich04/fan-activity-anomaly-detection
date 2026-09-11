# Fan and Activity Anomaly Detection

EE 446 lab work for on-device anomaly detection on an Arduino Nano 33 BLE Sense. The repository contains a prebuilt Edge Impulse fan-monitoring firmware image and source for a separate activity-anomaly detector using TensorFlow Lite for Microcontrollers.

## Fan monitoring firmware

The included report describes an Edge Impulse impulse using the Arduino Nano 33 BLE Sense Rev2 onboard accelerometer (`accX`, `accY`, and `accZ`). It uses a 2,000 ms window, 220 ms stride, and 100 Hz sampling rate. Spectral analysis produces 33 features, including RMS, spectral power, peak frequency, and peak height. The classifier has 33 inputs, dense layers of 32 and 16 units, and two outputs: `nominal` and `off`. A 32-cluster k-means anomaly detector is also described for normal-feature distributions.

`fan_monitoring/v10/` and `fan_monitoring/v11/` contain compiled firmware and platform-specific flash scripts. The Edge Impulse project source, fan mounting method, and raw fan recordings are not included in this checkout.

To flash the included fan firmware, connect a compatible Arduino Nano 33 BLE board and run the script for the operating system:

```powershell
fan_monitoring\v11\flash_windows.bat
```

```bash
bash fan_monitoring/v11/flash_linux.sh
# or
./fan_monitoring/v11/flash_mac.command
```

The scripts require `arduino-cli`, locate a connected Nano 33 BLE board, install `arduino:mbed_nano@4.0.2` when needed, and upload the binary from that directory.

## Activity anomaly detector

`TinyML_Lab6_Student_TODO.ipynb` trains an autoencoder on the included mHealth Subject 6 log. It uses left-ankle accelerometer columns, makes overlapping windows of 100 samples by three axes, and trains only the notebook's designated normal activity labels. The model is a 300-input autoencoder with dense layers of 32, 16, and 32 units before a 300-value reconstruction output. The notebook applies quantization-aware training and exports a full-int8 TFLite model.

`arduino/activity_anomaly/Activity_Anomaly.ino` embeds that model from `autoencoder_model.cc`. It reads three-axis acceleration through `Arduino_BMI270_BMM150`, samples on an approximately 50 Hz cadence, converts acceleration from g to m/s², quantizes each 100-sample window, and prints `Normal activity` or `Anomaly detected` from mean reconstruction error. The deployed threshold is `1.478182464838028`.

To rerun training, place the data file where the notebook expects it, then open the notebook:

```powershell
Copy-Item data\mHealth_subject6.log .\mHealth_subject6.log
jupyter notebook TinyML_Lab6_Student_TODO.ipynb
```

The notebook's first cell installs missing Python packages in its active kernel. To upload the activity detector, open `arduino/activity_anomaly/Activity_Anomaly.ino` in Arduino IDE with `autoencoder_model.cc` in the same sketch directory. The sketch includes TensorFlow Lite for Microcontrollers and `Arduino_BMI270_BMM150`.

## Hardware and tools

- Arduino Nano 33 BLE Sense Rev2 for the documented fan impulse
- Arduino Nano 33 BLE Sense-class board and onboard IMU for the activity sketch
- Edge Impulse firmware export and Arduino CLI for the fan binary
- TensorFlow, TensorFlow Model Optimization, TensorFlow Lite for Microcontrollers, Arduino IDE, and `Arduino_BMI270_BMM150` for the activity path

## Credits

Lab report author: Sparsh Dadhich. The mHealth dataset documentation credits Oresti Banos, Rafael Garcia, and Alejandro Saez, University of Granada. See `Dataset-README.txt` for the dataset citation and sensor documentation.
