#!/usr/bin/python3

import _thread
import time

# Define a function for the thread
def listen_to_incoming_data( threadName):
   

def run_tracking(threadName):
	#wait for incoming data, once data is recieved:
	#apply image filter to frame recieved
	#call calcOpticalFlowPyrLK on current frame and old frame
	#set current frame to old frame
	frame = cv2.imread(filepath, -1)
                frame_filtered = image_filter(frame, run_params)
    tracked_contour, _, _ = cv2.calcOpticalFlowPyrLK(
                        old_frame_filtered, frame_filtered, pts, None,
                        **lk_params)
# Create two threads as follows
try:
   _thread.start_new_thread( listen_to_incoming_data, ("wait_for_incoming",) )
   _thread.start_new_thread( listen_to_incoming_data, ("Thread-2",) )
except:
   print ("Error: unable to start thread")

while 1:
   pass