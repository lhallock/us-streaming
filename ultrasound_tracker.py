import socket
import cv2 as cv
import numpy as np
from struct import unpack, error
import time
import threading 
from enum import Enum
import os

from multisensorimport.tracking.image_proc_utils import get_filter_from_num
from multisensorimport.tracking.paramvalues import ParamValues

class DrawingState(Enum):
    """
    This class represents whether the user has started selecting points 
    along the muscle, and which stage of point selection they are at
    """
    STARTING_FIRST = 0
    DRAWING_FIRST = 1
    STARTING_SECOND = 2
    DRAWING_SECOND = 4
    DONE_SECOND = 5


IP = 'YOUR_IP_HERE'
RESET_DISTANCE = 200

class UltrasoundTracker:

    def __init__(self, muscle_thickness_file, image_directory):
        """
        Init method for UltrasoundTracker. This method starts a thread to
        recieve images from the ultrasound machine.

        Args:
            muscle_thickness_file: The filename to write the tracked thickness 
            values to
            image_directory: The folder prefix to save the ultrasound images in.
            Both raw and annotated images are saved.
        """
        self.THICKNESS_FILE = muscle_thickness_file
        self.IMAGE_DIRECTORY_RAW = image_directory + "_raw"
        self.IMAGE_DIRECTORY_FILTERED = image_directory + "_filtered"
        #imageMatrix is the recieved image from the ultrasound
        self.imageMatrix = np.zeros((326, 241), dtype = np.uint8)
        #boolean for if ultrasound data has been recieved yet
        self.got_data = False

        self.drawing = DrawingState.STARTING_FIRST
        self.old_frame_filtered = np.zeros((244,301), np.uint8)
        self.clicked_pts_set_one = []
        self.clicked_pts_set_two = []
        #create a thread to run recieve_data and start it
        t1 = threading.Thread(target=self.recieve_data) 
        t1.start()

    def recieve_data(self):
        """
        This function recieves the image from the ultrasound machine
        and writes them to imageMatrix
        """

        #set up socket to communicate with ultrasound machine
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((IP, 19001))
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
        self.imageMatrix = np.zeros((numberOfPoints, numberOfLines), dtype = np.uint8)
        recieved = 0
        buffer = b''
        while (recieved != header[8]): 
            buffer += conn.recv(header[8] - recieved)
            recieved = len(buffer)
        nparr = np.frombuffer(buffer, np.uint8)
        for j in range(numberOfLines):
            for i in range(numberOfPoints):
                self.imageMatrix[i][j] = nparr[i+ j *(numberOfPoints + 8)]
        iteration += 1
        self.got_data = True

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
            for j in range(numberOfLines):
                for i in range(numberOfPoints):
                    self.imageMatrix[i][j] = nparr[i+ j *(numberOfPoints + 8)]
            #if we have not recieved an image, break
            if not data:
                break
            iteration += 1
            
    def collect_clicked_pts(self, event, x, y, flags, param):
        """ 
        This function stores the first 10 points the user clicks on
        in clicked_pts_set_one, and the next 10 points in clicked_pts_set_two. 
        """

        if event == cv.EVENT_LBUTTONDOWN:
            if self.drawing == DrawingState.STARTING_FIRST or self.drawing == DrawingState.DRAWING_FIRST:
                self.drawing = DrawingState.DRAWING_FIRST
                self.clicked_pts_set_one.append((x,y))
                if(len(self.clicked_pts_set_one) >= 10):
                    self.drawing = DrawingState.STARTING_SECOND
                    print("Start drawing second line now!", flush=True)
            elif self.drawing == DrawingState.STARTING_SECOND or self.drawing == DrawingState.DRAWING_SECOND:
                self.drawing = DrawingState.DRAWING_SECOND
                self.clicked_pts_set_two.append((x,y))
                if(len(self.clicked_pts_set_two) >= 10):
                    self.drawing = DrawingState.DONE_SECOND
            elif self.drawing == DrawingState.DONE_SECOND:
                self.reset_points()

                
    def extract_contour_pts(self, img):
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
        frame_HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        frame_threshold = cv.inRange(frame_HSV, (0, 255, 255), (20, 255, 255))
        contours, _ = cv.findContours(frame_threshold, cv.RETR_TREE,
                                       cv.CHAIN_APPROX_SIMPLE)
        # convert largest contour to tracking-compatible numpy array
        points = []

        for j in range(len(contours)):
            for i in range(len(contours[j])):
                points.append(np.array(contours[j][i], dtype=np.float32))

        np_points = np.array(points)

        return np_points

    def reset_points(self):
        """ 
        Resets self.points_set_one and self.points_set_two to the original tracked
        points 
        """ 
        self.points_set_one = self.original_points_set_one.copy()
        self.points_set_two = self.original_points_set_two.copy()

    def main(self, pipe):
        """ 
        This method is started as a thread by start_process.py. It first allows the user 
        to select two areas to track, then runs optical flow tracking on two sets 
        of points on the muscle. It records the vertical muscle thickness
        as the vertical distance between the means of these two clusters of points.
        It also sends the thickness to the graphing program, and saves every 10'th image. 

        Args:
            pipe: the pipe that is connected to the graphing program
        """
        with open(self.THICKNESS_FILE, "w") as thickness_file:
            thickness_file.write("Muscle thickness data\n")
        #create opencv window to display image
        cv.namedWindow('image')
        cv.setMouseCallback('image',self.collect_clicked_pts)


        #set up variables for tracking
        first_loop = True
        run_params = ParamValues()
        window_size = run_params.LK_window
        lk_params = dict(winSize=(window_size, window_size),
                         maxLevel=run_params.pyr_level,
                         criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT,
                                   10, 0.03))
        image_filter = get_filter_from_num(3)

        # Green color in BGR 
        color = (0, 0, 255) 
          
        # Line thickness of 4 px 
        thickness = 4

        self.points_set_one = None
        self.points_set_two = None
        counter = 0

        while 1:
            if self.got_data:
                #resize imageMatrix so it has a larger width than height
                resized = cv.resize(self.imageMatrix, (int(1.25*self.imageMatrix.shape[1]), (int(.75*self.imageMatrix.shape[0]))), interpolation 
                    = cv.INTER_AREA).copy()
                if self.drawing != DrawingState.DONE_SECOND:
                    old_frame = resized.copy()
                    old_frame_color = cv.cvtColor(old_frame,
                                                   cv.COLOR_GRAY2RGB).copy()
                    # visualize 
                    contour_image = cv.polylines(old_frame_color, 
                                          [np.array(self.clicked_pts_set_one).reshape((-1, 1, 2)), np.array(self.clicked_pts_set_two).reshape((-1, 1, 2))],  
                                          False, color,  
                                          thickness).copy()
                    cv.imshow('image', contour_image)

                    cv.waitKey(1)

                elif self.drawing == DrawingState.DONE_SECOND and self.points_set_one is None and self.points_set_two is None:
                    old_frame = resized.copy()
                    old_frame_color = cv.cvtColor(old_frame,
                                                   cv.COLOR_GRAY2RGB).copy()

                    #draw two polygons around the two sets of selected points and use extract_contour_pts to get two good sets of points to track
                    contour_image = cv.polylines(old_frame_color.copy(), [np.array(self.clicked_pts_set_one).reshape((-1, 1, 2))],  
                                              False, color,
                                              thickness).copy()
                    self.points_set_one = self.extract_contour_pts(contour_image)
                    self.original_points_set_one = self.points_set_one.copy()

                    contour_image = cv.polylines(old_frame_color.copy(), [np.array(self.clicked_pts_set_two).reshape((-1, 1, 2))],  
                                              False, color,
                                              thickness).copy()
                    self.points_set_two = self.extract_contour_pts(contour_image)
                    self.original_points_set_two = self.points_set_two.copy()

                # track and display specified points through images
                # if it's the first image, we will just set the old_frame varable
                elif first_loop:

                    old_frame = resized.copy()

                    # apply filters to frame
                    self.old_frame_filtered = image_filter(old_frame, run_params)

                    first_loop = False

                    cv.waitKey(1)

                else:
                    # read in new frame
                    frame = resized.copy()
                    frame_filtered = image_filter(frame, run_params)

                    #perform optical flow tracking to track where points went between image frames
                    tracked_contour_one, _, _ = cv.calcOpticalFlowPyrLK(
                        self.old_frame_filtered, frame_filtered, self.points_set_one, None,
                        **lk_params)

                    tracked_contour_two, _, _ = cv.calcOpticalFlowPyrLK(
                        self.old_frame_filtered, frame_filtered, self.points_set_two, None,
                        **lk_params)

                    tracked_contour_one = tracked_contour_one.reshape((-1, 2))
                    tracked_contour_two = tracked_contour_two.reshape((-1, 2))


                    # update for next iteration
                    self.old_frame_filtered = frame_filtered.copy()

                    self.points_set_one = tracked_contour_one.copy()
                    self.points_set_two = tracked_contour_two.copy()

     
                    frame_color = cv.cvtColor(frame_filtered,
                                                   cv.COLOR_GRAY2RGB).copy()
 
                    #calculate average distance to center of clusters, and reset if too large
                    mean_one = tuple(np.mean(tracked_contour_one, axis = 0, dtype=np.int))
                    mean_two = tuple(np.mean(tracked_contour_two, axis = 0, dtype = np.int))

                    sum_distances_one, sum_distances_two = 0, 0

                    for i in range(len(tracked_contour_one)):
                        x, y = tracked_contour_one[i].ravel()
                        cv.circle(frame_color, (int(x), int(y)), 3, (0, 0, 255), -1)
                        sum_distances_one += (x - mean_one[0])**2 + (y - mean_one[1])**2

                    for i in range(len(tracked_contour_two)):
                        x, y = tracked_contour_two[i].ravel()
                        cv.circle(frame_color, (int(x), int(y)), 3, (255, 0, 0), -1)
                        sum_distances_two += (x - mean_two[0])**2 + (y - mean_two[1])**2

                    average_distance_set_one = float(sum_distances_one)/len(tracked_contour_one)
                    average_distance_set_two = float(sum_distances_two)/len(tracked_contour_two)
                    max_average_distance =  max(average_distance_set_one, average_distance_set_two)
                    if max_average_distance > RESET_DISTANCE:
                        self.reset_points()
                        print("average squared distance was ", max_average_distance)
                        print("resetting points!", flush=True)
                        continue

                    #draw line representing thickness
                    cv.line(frame_color, mean_one, mean_two, (255, 0, 255), 3)
                    middle_x = int((mean_one[0] + mean_two[0])/2)
                    topmost_y = max(mean_one[1], mean_two[1])
                    bottommost_y = min(mean_one[1], mean_two[1])
                    cv.line(frame_color, (middle_x - 10, topmost_y), (middle_x + 10, topmost_y), (0, 255, 0), 3)
                    cv.line(frame_color, (middle_x - 10, bottommost_y), (middle_x + 10, bottommost_y), (0, 255, 0), 3)
                    vertical_distance = topmost_y - bottommost_y

                    now = time.time()
                    str_now = str(now)

                    #send data to graphing program, and save every 10'th image
                    pipe.send(vertical_distance)

                    if counter == 10:
                        cv.imwrite(os.path.join(os.getcwd(), self.IMAGE_DIRECTORY_RAW, str_now) + ".jpg", resized)
                        cv.imwrite(os.path.join(os.getcwd(), self.IMAGE_DIRECTORY_FILTERED, str_now) + ".jpg", frame_color)
                        with open(self.THICKNESS_FILE, "a") as thickness_file:
                            thickness_file.write(str_now + ": " + str(vertical_distance) + "\n")
                        counter = 0
                    
                    cv.imshow('image', frame_color)

                    # wait 1ms 
                    cv.waitKey(1)
                    counter += 1
