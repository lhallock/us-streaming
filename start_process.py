"""Script to run ultrasound tracking code and tracker.

Example:
    Run this function via

        $ python start_process.py <trial_num> <thickness_file> <images_folders_prefix>

    for
        trial_num: integer value corresponding to desired plotting/recording
            0: records and plots ultrasound-measured thickness, sEMG-measured
                activation, and force trajectories
            1: records thickness, activation, and force, but plots only force
            2: records thickness, activation, and force, but plots only
                thickness
            3: records thickness, activation, and force, but plots only
                activation
            4: records and plots only ultrasound-measured thickness (sEMG and
            force sensors need not be set up)
        thickness_file: file to which time series thickness values should be
            saved (e.g., 'thickness.txt')
        images_folders_prefix: prefix of folder names to which to save
            processed and unprocessed images (e.g., 'images' for preexisting
            folders 'images_filtered' and 'images_raw')
"""
import multiprocessing as mp
import sys

from amg_emg_force_control.grapher_game.game.main import GraphingMain
from ultrasound_tracker import UltrasoundTracker

sys.path.append('../')

if __name__ == '__main__':
    """Starts up ultrasound tracking code and grapher in separate processes."""
    trial_number = sys.argv[1]
    ultrasound_muscle_thickness_file = sys.argv[2]
    ultrasound_image_directory = sys.argv[3]
    TRIAL_FILENAME = "../amg_emg_force_control/grapher_game/test_scripts/trial_" + str(
        trial_number) + ".txt"

    p_out, p_in = mp.Pipe()
    p1 = UltrasoundTracker(ultrasound_muscle_thickness_file,
                           ultrasound_image_directory)
    p2 = GraphingMain()

    # start p2 as another process
    p2 = mp.Process(target=p2.main,
                    args=(TRIAL_FILENAME, ["serial_port=COM8"], p_out))
    p2.start()

    p1.main(p_in)
    p2.join()
