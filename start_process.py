import multiprocessing as mp
from ultrasound_tracker import UltrasoundTracker
import sys
sys.path.append('../')
from amg_emg_force_control.grapher_game.game.main import GraphingMain

if __name__ == '__main__':
	"""  Starts up ultrasound tracking code and grapher, in separate processes. """
	trial_number = sys.argv[1]
	muscle_thickness_file = sys.argv[2]
	image_directory = sys.argv[3]
	FILENAME= "../amg_emg_force_control/grapher_game/test_scripts/trial_" + str(trial_number) + ".txt"

	p_out, p_in = mp.Pipe()
	p1  = UltrasoundTracker(muscle_thickness_file, image_directory)
	p2 = GraphingMain()

	# start p2 as another process
	p2 = mp.Process(target=p2.main, args=(FILENAME, ["serial_port=COM8"], p_out))
	p2.start()     

	p1.main(p_in)
	p2.join() 