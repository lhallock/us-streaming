"""Code to stream ultrasound images and extract muscle thickness via optical
flow.

This library contains code to receive ultrasound images from the eZono 4000
ultrasound machine, perform optical flow tracking on two user-selected areas of
the muscle to determine time-varying muscle thickness, save these ultrasound
images and thickness values, and start up an external process to graph this
thickness in real time alongside additional sensor values and guiding
trajectories.
"""
__version__ = '0.0.0'
__author__ = 'Bhavna Sud and Laura A. Hallock'
