# Toward  Real-Time  Muscle  Force  Inference  and  Device  Control  via Optical-Flow-Tracked  Muscle  Deformation

This repo contains code used to 
- collect and visualize ultrasound muscle time series data in real time; and
- perform real time optical flow tracking of muscle deformation (i.e., contour motion) from time series ultrasound frames.

**NOTE**: This (`master`) branch of this code contains the most recent stable version of our tracking and analysis code; it does not contain code for publications currently under review, and may have been updated since submitting the publication for review. To access the version of the codebase currently under review, please visit the `tnsre-2021` branch [here](https://github.com/lhallock/us-streaming/tree/tnsre-2021).

## Installation

### Downloading this repository

To download all modules and scripts, clone this repository via

```bash
git clone https://github.com/lhallock/us-streaming.git
```

### Dependencies

In order to run this code, you will need access to an Ezono Ultrasound machine (https://www.ezono.com/en/ezono-5000/) that you can ssh into.

Our expiriment was also run with EMG and Force sensors, which you can set up according to https://github.com/cmitch/amg_emg_force_control.

To run the code, the following Python modules are required, all of which can be installed via `pip`: 
`numpy`, `opencv-python`, `future`, `iso8601`, `numpy`, `PyQt5`, `PyQt5-sip`, `pyqtgraph`, `pyserial`, `PyYAML`, and `serial`

It is also required that the following repository: https://github.com/cmitch/amg_emg_force_control is installed in the same directory as this repository.

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
