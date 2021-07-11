import multiprocessing as mp
from ultrasound_tracker import UltrasoundTracker
import sys
sys.path.append('../')
from amg_emg_force_control.grapher_game.game.main import GraphingMain

if __name__ == '__main__':
	"""  
	Starts up ultrasound tracking code and grapher, in separate processes. 

	Args:
		trial_number: 
			In all trials, Ultrasound, EMG, and Force are recorded and saved,
			but each trial varies in which ones are displayed on the grapher
			0: Graph Ultrasound, EMG, and Force
			1: Graph Force
			2: Graph Ultrasound
			3: Graph EMG
		ultrasound_muscle_thickness_file: The file to save the tracked ultasound
		thickness to
		ultrasound_image_directory: The folder prefix to save the ultrasound images to
	 """

	trial_number = sys.argv[1]
	ultrasound_muscle_thickness_file = sys.argv[2]
	ultrasound_image_directory = sys.argv[3]
	TRIAL_FILENAME= "../amg_emg_force_control/grapher_game/test_scripts/trial_" + str(trial_number) + ".txt"

	p_out, p_in = mp.Pipe()
	p1  = UltrasoundTracker(ultrasound_muscle_thickness_file, ultrasound_image_directory)
	p2 = GraphingMain()

	# start p2 as another process
	p2 = mp.Process(target=p2.main, args=(TRIAL_FILENAME, ["serial_port=COM8"], p_out))
	p2.start()     

	p1.main(p_in)
	p2.join() 