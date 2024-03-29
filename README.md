# Tracking Muscle Deformation via Optical Flow in Time Series Ultrasound Images

![us-streaming tracking exemplar](header.gif)

This repo contains code used to 
- perform real time optical flow tracking of muscle deformation (i.e., contour motion) from time series ultrasound frames; and
- record and visualize this data.

**If you use this code for academic purposes, please cite the following publication**: Laura A. Hallock, Bhavna Sud, Chris Mitchell, Eric Hu, Fayyaz Ahamed, Akash Velu, Amanda Schwartz, and Ruzena Bajcsy, "[Toward Real-Time Muscle Force Inference and Device Control via Optical-Flow-Tracked Muscle Deformation](https://ieeexplore.ieee.org/document/9641847)," in _IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)_, IEEE, 2021.

This README primarily describes the methods needed to replicate the data collection procedure used in the publication above. The code and documentation are provided as-is; however, we invite anyone who wishes to adapt and use it under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Installation

### Downloading this repository

To download all modules and scripts, clone this repository via

```bash
git clone https://github.com/lhallock/us-streaming.git
```

and navigate to this branch via

```bash
git checkout tnsre-2021
```

### Dependencies

This code is designed for use with an [eZono 4000](https://www.ezono.com/en/) ultrasound machine with SSH access, though it can likely be adapted to other platforms.

This code is designed for use alongside our time series visualization and recording code, which can be found [here](https://github.com/cmitch/amg_emg_force_control) and enables simultaneous collection of ultrasound, surface electromyography (sEMG), force, and other data. **Both repositories must be downloaded into the same directory for this code to run without modification.**

To run the code, the following Python modules are required, all of which can be installed via `pip`: `numpy`, `opencv-python`, `future`, `iso8601`, `PyQt5`, `PyQt5-sip`, `pyqtgraph`, `pyserial`, `PyYAML`, and `serial`.

---

## Ultrasound muscle thickness tracking \& graphing

This section describes the file structure and code necessary to run ultrasound tracking. Two main scripts are included: the first, [`start_process.py`](start_process.py), starts the graphing code from the [streaming repository](https://github.com/cmitch/amg_emg_force_control) as a separate process and runs the ultrasound tracking code in its own process; the second, [`ultrasound_tracker.py`](ultrasound_tracker.py), starts one thread that receives ultrasound images from the eZono and another that runs optical flow tracking on two user selected areas of the muscle to determine muscle thickness, then saves these thickness values and the received ultrasound images and sends the thickness values to the graphing process.

### Setup

The `us-streaming` and `amg_emg_force_control` repositories should be arranged as follows, with empty files and folders manually created as listed:

```bash
├── amg_emg_force_control # visualization/recording repository linked above
├── us-streaming # this repository
│   ├── images_filtered # empty folder
│   ├── images_raw # empty folder
│   ├── thickness.txt # empty file
│   ├── ...
```

Note that the names `images_filtered`, `images_raw`, and `thickness.txt` are arbitrary and can be modified, as long as they're also changed in the commands below.

Next, setup a python virtual environment in a new directory by running 
```bash
python -m venv DIR_NAME
```
then navigate to the amg_emg_force_control repo and run
```bash
pip install -r requirements.txt
pip install -e .
```

### Usage

Steps:
1. Go to the us-streaming folder
2. Inside us-streaming, create two folders called images_raw and images_filtered
3. In terminal, go to us-streaming and type 
```bash
python start_process.py <trial_num> <thickness_file> <images_folders_prefix>
```
specifying the above command line arguments as follows:
- `trial_num`: which trial you want to run
  - `0`: Graphs/records Ultrasound, EMG, and Force
  - `1`: Records Ultrasound, EMG, and Force, but only graphs Force
  - `2`: Records Ultrasound, EMG, and Force, but only graphs Ultrasound
  - `3`: Records Ultrasound, EMG, and Force, but only graphs EMG
  - `4`: Records and graphs only Ultrasound
- `thickness_file`: filename to save ultrasound tracked muscle thickness to 
- `images_folders_prefix`: the prefix for the name of the folders to save the ultrasound images to (images will be the two folders [images_folders_prefix]_raw and [images_folders_prefix]_filtered.) Make sure these folders have been created before running this command.

For trial 4, you can run:
python start_process.py 4 thickness.txt images

4. Ssh into the ultrasound and type b-mode-compounded-data-out <YOUR_COMPUTER_IP> 19001, replacing <YOUR_COMPUTER_IP> with the IP address of the computer you are running this code on.
5. On the displayed image, select 10 dots on the top of the muscle and 10 dots on the bottom
6. Enter in a filename you want to save the recording to. Wait until the green line in the graph reaches the end, then press record.
7. Min and max calibrate
8. Click start trajectory 0, and run the trial
9. Click stop recording
10. Control-c out of everything

The images will be saved in us-streaming/images_raw and us-streaming/images_filtered. The thickness will be saved in us-streaming/thickness.txt. The recorded graph will be saved in /data/.

---

## Visualizing \& analyzing generated time series

The procedure above generates JPEG images and corresponding time series text files that are accessible via any standard image viewer and text editor, respectively. The generated Python `*.p` archive files, which contain time series data for all streams and associated metadata, can be accessed via our corresponding [analysis repository](https://github.com/lhallock/openarm-multisensor/tree/tnsre-dev).

---

## Contributing

If you're interested in contributing or collaborating, please reach out to `lhallock [at] eecs.berkeley.edu`. 

