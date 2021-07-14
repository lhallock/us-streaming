# Tracking Muscle Deformation via Optical Flow in Time Series Ultrasound Images

This repo contains code used to 
- perform real time optical flow tracking of muscle deformation (i.e., contour motion) from time series ultrasound frames; and
- record and visualize this data.

**If you use this code for academic purposes, please cite the following publication**: Laura A. Hallock, Bhavna Sud, Chris Mitchell, Eric Hu, Fayyaz Ahamed, Akash Velu, Amanda Schwartz, and Ruzena Bajcsy, "[Toward Real-Time Muscle Force Inference and Device Control via Optical-Flow-Tracked Muscle Deformation](https://people.eecs.berkeley.edu/~lhallock/publication/hallock2021tnsre/)," in _IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)_, IEEE, 2021. (under review)

**NOTE**: This code branch has been updated in preparation for the paper submission above, currently under review. To access the latest stable code release, visit the `master` branch [here](https://github.com/lhallock/us-streaming/).

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

This code is designed for use alongside our time series visualization and recording code, which can be found [here](https://github.com/cmitch/amg_emg_force_control) and enables simultaneous collection of ultrasound, surface electromyography (sEMG), force, and other data. Both repositories must be downloaded into the same directory for this code to run without modification.

To run the code, the following Python modules are required, all of which can be installed via `pip`: `numpy`, `opencv-python`, `future`, `iso8601`, `PyQt5`, `PyQt5-sip`, `pyqtgraph`, `pyserial`, `PyYAML`, and `serial`.

---

## Ultrasound muscle thickness tracking and graphing

This section describes the file structure and code necessary to run ultrasound tracking. Two main scripts are included: the first, [`start_process.py`](start_process.py), starts the graphing code from https://github.com/cmitch/amg_emg_force_control as a separate process, and runs the ultrasound tracking code in its own process; the second, [`ultrasound_tracker.py`](ultrasound_tracker.py), starts one thread that recieves ultrasound images from the eZono, and another thread that runs optical flow tracking on two user selected areas of the muscle to determine muscle thickness, saves these thickness values and the recieved ultrasound images, and sends the thickness values to the graphing process.

### Setup

The us-streaming and amg_emg_force_control repositories should be arranged as follows.
```bash
├── amg_emg_force_control
├── us-streaming
│   ├── images_filtered (empty folder)   
│   ├── images_raw (empty folder)
│   ├── thickness.txt (empty file)
│   ├── ...
```

You can rename images_filtered, images_raw, and thickness.txt to anything you want, but be sure to change these names in the commands below.

### Usage

To run Trials 0-3, set up EMG/Force sensors as described in https://github.com/cmitch/amg_emg_force_control. 
For just ultrasound, you can run Trial 4. 

Steps:
1. With terminal or git bash, go to the amg_emg_force_control folder, and type amg && amg_env
3. Go to the ../us-streaming folder
4. Inside us-streaming, create two folders called images_raw and images_filtered
4. In terminal, go to us-streaming and type 
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

5. Ssh into the ultrasound and type b-mode-compounded-data-out <YOUR_COMPUTER_IP> 19001, replacing <YOUR_COMPUTER_IP> with the IP address of the computer you are running this code on.
6. On the displayed image, select 10 dots on the top of the muscle and 10 dots on the bottom
7. Enter in a filename you want to save the recording to. Wait until the green line in the graph reaches the end, then press record.
8. Min and max calibrate
9. Click start trajectory 0, and run the trial
10. Click stop recording
11. Control-c out of everything

The images will be saved in us-streaming/images_raw and us-streaming/images_filtered. The thickness will be saved in us-streaming/thickness.txt. The recorded graph will be saved in /data/.

## Contributing

If you're interested in contributing or collaborating, please reach out to `lhallock [at] eecs.berkeley.edu`. 

