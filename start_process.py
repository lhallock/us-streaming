import multiprocessing as mp
import sys

from amg_emg_force_control.grapher_game.game.main import GraphingMain
from ultrasound_tracker import UltrasoundTracker

sys.path.append('../')

if __name__ == '__main__':
    """  
	Starts up ultrasound tracking code and grapher, in separate processes. 

	Args:
		trial_number: 
			In all trials except trial 4, Ultrasound, EMG, and Force are recorded and saved,
			but each of these trials varies in which ones are displayed on the grapher
			0: Graph Ultrasound, EMG, and Force
			1: Graph Force
			2: Graph Ultrasound
			3: Graph EMG
			4: Graph Ultrasound (EMG and Force do not need to be set up)
		ultrasound_muscle_thickness_file: The file to save the tracked ultasound
		thickness to
		ultrasound_image_directory: The folder prefix to save the ultrasound images to
	 """

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
