from multiprocessing import Process, Pipe
from socketpython import SocketPython
import sys
sys.path.append('../')

from amg_emg_force_control import *
from amg_emg_force_control.grapher_game.game.main import GraphingMain
FILENAME= "../amg_emg_force_control/grapher_game/test_scripts/ultrasound_emg_trial.txt"

if __name__ == '__main__':
    p_out, p_in = Pipe()
    p1  = SocketPython()
    p2 = GraphingMain()

    # start p2 as another process
    p2 = Process(target=p2.main, args=(FILENAME, ["serial_port=COM8"], p_out))
    # p2.daemon = True
    p2.start()     # Launch the stage2 process

    p1.main(p_in)
    p2.join() 