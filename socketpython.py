import socket
import cv2 as cv
import numpy as np
from struct import *
import time
import threading 
from enum import Enum

from multisensorimport.tracking import supporters_utils
from multisensorimport.tracking.image_proc_utils import *
from multisensorimport.tracking.point_proc_utils import *
from multisensorimport.tracking.paramvalues import *

class DrawingState(Enum):
    STARTING_FIRST = 0
    DRAWING_FIRST = 1
    STARTING_SECOND = 2
    DRAWING_SECOND = 4
    DONE_SECOND = 5

#imageMatrix is the recieved image from the ultrasound
imageMatrix = np.zeros((326, 241), dtype = np.uint8)
#boolean for if ultrasound data has been recieved yet
got_data = False

drawing = DrawingState.STARTING_FIRST
old_frame_filtered = np.zeros((244,301), np.uint8)
clicked_pts_set_one = []
clicked_pts_set_two = []

"""This function recieves the image from the ultrasound machine and writes them to imageMatrix"""
def recieve_data():
    #define global variables so we can modify them in this function
    global imageMatrix
    global got_data

    #set up socket to communicate with ultrasound machine
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('192.168.1.151', 19001))
    s.listen(1)
    conn, addr = s.accept()
    print(addr)

    #first iteration to set imageMatrix size
    iteration = 0
    recieved = 0
    while (recieved != 100):
        data = conn.recv(100 - recieved)
        recieved += len(data)
    header = unpack('IIIIIIIIIQIIIIIIIIIIIII', data)
    numberOfLines = header[13]
    numberOfPoints = header[14]
    imageMatrix = np.zeros((numberOfPoints, numberOfLines), dtype = np.uint8)
    recieved = 0
    buffer = b''
    while (recieved != header[8]): 
        buffer += conn.recv(header[8] - recieved)
        recieved = len(buffer)
    nparr = np.frombuffer(buffer, np.uint8)
    for j in range(numberOfLines):
        for i in range(numberOfPoints):
            imageMatrix[i][j] = nparr[i+ j *(numberOfPoints + 8)]
    iteration += 1
    got_data = True
    #rest of iterations
    while 1:
        #recieve header data 
        recieved = 0
        while (recieved != 100):
            data = conn.recv(100 - recieved)
            recieved += len(data)
        #unpack header data so we can access it
        try:
            header = unpack('IIIIIIIIIQIIIIIIIIIIIII', data)
        except error:
            print("unpack error")
        #recieved image size will be numberOfLines * numberOfPoints
        numberOfLines = header[13]
        numberOfPoints = header[14]

        #recieve image data
        recieved = 0
        buffer = b''
        while (recieved != header[8]): 
            buffer += conn.recv(header[8] - recieved)
            recieved = len(buffer)
        nparr = np.frombuffer(buffer, np.uint8)

        #fill in imageMatrix based on recieved data
        if iteration % 10 == 0:
            for j in range(numberOfLines):
                for i in range(numberOfPoints):
                    imageMatrix[i][j] = nparr[i+ j *(numberOfPoints + 8)]
        #if we have not recieved an image, break
        if not data:
            break
        iteration += 1
    #close socket connection once we're done
    conn.close

        
def draw_polygon(event, x, y, flags, param):

    global clicked_pts_set_one, clicked_pts_set_two, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        if drawing == DrawingState.STARTING_FIRST or drawing == DrawingState.DRAWING_FIRST:
            drawing = DrawingState.DRAWING_FIRST
            clicked_pts_set_one.append((x,y))
            if(len(clicked_pts_set_one) >= 10):
                drawing = DrawingState.STARTING_SECOND
                print("START DRAWING SECOND LINE NOW!")
        elif drawing == DrawingState.STARTING_SECOND or drawing == DrawingState.DRAWING_SECOND:
            drawing = DrawingState.DRAWING_SECOND
            clicked_pts_set_two.append((x,y))
            if(len(clicked_pts_set_two) >= 10):
                drawing = DrawingState.DONE_SECOND
            
def extract_contour_pts(img):
    """Extract points from largest contour in PGM image.

    This function is used to extract ordered points along the largest detected
    contour in the provided PGM image and format them for use by OpenCV image
    tracking. In particular, this function is used to extract the fascial
    border of the brachioradialis muscle in a mask manually segmented from a
    given ultrasound frame. It is typically used to initialize points to track.

    Args:
        img: cv image with contour drawn

    Returns:
        numpy.ndarray of contour points
    """
    # convert image to grayscale if it is color
    # if len(img.shape) > 2:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold_level = 100
    
    # # binarize image
    # _, binarized = cv2.threshold(img, threshold_level, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image', binarized)
    # cv2.waitKey(0)
    frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (0, 255, 255), (20, 255, 255))
    # flip image (need a white object on black background)
    # flipped = cv2.bitwise_not(binarized)
    # print("flipped is ",flipped)
    contours, _ = cv2.findContours(frame_threshold, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # convert largest contour to tracking-compatible numpy array
    points = []
    for j in range(len(contours)):
        for i in range(len(contours[j])):
            points.append(np.array(contours[j][i], dtype=np.float32))

    np_points = np.array(points)

    return np_points

def main():
    global old_frame_filtered, drawing, clicked_pts_set_one, clicked_pts_set_two
    #create a thread to run recieve_data and start it
    t1 = threading.Thread(target=recieve_data) 
    t1.start()

    #create opencv window to display image
    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_polygon)


    #set up variables for tracking
    first_loop = True
    enter_pressed = False
    run_params = ParamValues()
    pts = np.array([100, 100]).astype(np.float32).reshape((1, 1, 2))
    filter_type = 3
    window_size = run_params.LK_window
    lk_params = dict(winSize=(window_size, window_size),
                     maxLevel=run_params.pyr_level,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    image_filter = get_filter_from_num(filter_type)

    isClosed = False
  
    # Green color in BGR 
    color = (0, 0, 255) 
      
    # Line thickness of 8 px 
    thickness = 4

    points_set_one = None
    points_set_two = None
    while 1:
        if got_data:
            # print("got data!")
            #resize imageMatrix so it has a larger width than height
            resized = cv.resize(imageMatrix, (int(1.25*imageMatrix.shape[1]), (int(.75*imageMatrix.shape[0]))), interpolation 
                = cv.INTER_AREA).copy()
            if drawing != DrawingState.DONE_SECOND:
                # print('loop again')
                old_frame = resized.copy()
                # old_frame_filtered = image_filter(old_frame, run_params)
                old_frame_color = cv2.cvtColor(old_frame,
                                               cv2.COLOR_GRAY2RGB).copy()
                # visualize 
                
                # Using cv2.polylines() method 
                # Draw a Green polygon with  
                # thickness of 1 px 
                contour_image = cv2.polylines(old_frame_color, [np.array(clicked_pts_set_one).reshape((-1, 1, 2)), np.array(clicked_pts_set_two).reshape((-1, 1, 2))],  
                                      isClosed, color,  
                                      thickness).copy()
                cv.imshow('image', contour_image)
                # else:
                #     cv.imshow('image', old_frame_color)

                key = cv.waitKey(1)

            elif drawing == DrawingState.DONE_SECOND and points_set_one is None and points_set_two is None:
                old_frame = resized.copy()
                old_frame_color = cv2.cvtColor(old_frame,
                                               cv2.COLOR_GRAY2RGB).copy()
                # print("clicked pts set one", clicked_pts_set_one)
                contour_image = cv2.polylines(old_frame_color.copy(), [np.array(clicked_pts_set_one).reshape((-1, 1, 2))],  
                                          isClosed, color,
                                          thickness).copy()
                points_set_one = extract_contour_pts(contour_image)
                # print("points set one", points_set_one)

                contour_image = cv2.polylines(old_frame_color.copy(), [np.array(clicked_pts_set_two).reshape((-1, 1, 2))],  
                                          isClosed, color,
                                          thickness).copy()
                # print("clicked_pts_set_two", clicked_pts_set_two)
                points_set_two = extract_contour_pts(contour_image)
                # print("points_set_two", points_set_two)


            # track and display specified points through images
            # if it's the first image, we will just set the old_frame varable
            elif first_loop:

                old_frame = resized.copy()

                # apply filters to frame
                old_frame_filtered = image_filter(old_frame, run_params)

                first_loop = False

                key = cv.waitKey(1)

            else:
                # read in new frame
                frame = resized.copy()
                frame_filtered = image_filter(frame, run_params)

                #perform optical flow tracking to track where points went between image frames
                tracked_contour_one, _, _ = cv.calcOpticalFlowPyrLK(
                    old_frame_filtered, frame_filtered, points_set_one, None,
                    **lk_params)

                tracked_contour_two, _, _ = cv.calcOpticalFlowPyrLK(
                    old_frame_filtered, frame_filtered, points_set_two, None,
                    **lk_params)

                tracked_contour_one = tracked_contour_one.reshape((-1, 2))
                tracked_contour_two = tracked_contour_two.reshape((-1, 2))


                # update for next iteration
                old_frame_filtered = frame_filtered.copy()
                points_set_one = tracked_contour_one.copy()
                points_set_two = tracked_contour_two.copy()

 
                frame_color = cv2.cvtColor(frame_filtered,
                                               cv2.COLOR_GRAY2RGB).copy()
                # visualize 
                # draw the tracked contour
                for i in range(len(tracked_contour_one)):
                    x, y = tracked_contour_one[i].ravel()
                    cv.circle(frame_color, (x, y), 3, (0, 0, 255), -1)
                mean_one = tuple(np.mean(tracked_contour_one, axis = 0))

                for i in range(len(tracked_contour_two)):
                    x, y = tracked_contour_two[i].ravel()
                    cv.circle(frame_color, (x, y), 3, (255, 0, 0), -1)
                mean_two = tuple(np.mean(tracked_contour_two, axis = 0))

                distance = [mean_two[0] - mean_one[0], mean_two[1] - mean_one[1]]
                muscle_thickness = np.linalg.norm(distance)
                file = open("thickness.txt", "w") 
                file.write(str(muscle_thickness))
                file.close()

                cv.line(frame_color, mean_one, mean_two, (255, 0, 255), 3)
                
                cv.imshow('image', frame_color)

                key = cv.waitKey(1)

    if viz:
        cv.destroyAllWindows()
"""This is the main method. It creates a thread to run recieve_data, and 
then continually loops and displays the recieved image. I am also working on adding LK tracking."""
if __name__ == "__main__":
    main()


        



