from multiprocessing import Process, Pipe
from socketpython import SocketPython
import sys
sys.path.append('../')

from amg_emg_force_control2 import *
from amg_emg_force_control2.grapher_game.game.main import GraphingMain
FILENAME= "../amg_emg_force_control2/grapher_game/test_scripts/test_and_ultrasound.txt"

if __name__ == '__main__':
    p_out, p_in = Pipe()
    p1  = SocketPython()
    p2 = GraphingMain()

    # start p2 as another process
    p2 = Process(target=p2.main, args=(FILENAME, None, p_out))
    # p2.daemon = True
    p2.start()     # Launch the stage2 process

    p1.main(p_in)
    p2.join() 