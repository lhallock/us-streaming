import multiprocessing as mp
from socketpython import SocketPython
import sys
sys.path.append('../')

from amg_emg_force_control import *
from amg_emg_force_control.grapher_game.game.main import GraphingMain
FILENAME= "../amg_emg_force_control/grapher_game/test_scripts/ultrasound_emg_trial.txt"

if __name__ == '__main__':
	shared = mp.Value("f", 0)
	p1  = SocketPython()
	p2 = GraphingMain()

	# start p2 as another process
	p2 = mp.Process(target=p2.main, args=(FILENAME, ["serial_port=COM8"], shared))
	p2.start()     

	p1.main(shared)
	p2.join() 