#!/usr/bin/env python3

# import the necessary libraries
import argparse
import cv2
import imutils
import json
import logging
import numpy as np
import time
import wpilib

from cscore import CameraServer
from imutils.video import FPS, VideoStream
from networktables import NetworkTables
from networktables import NetworkTablesInstance
from scipy.interpolate import interp1d

def main():
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--width", type=int, default=640, help="the frame width")
    ap.add_argument("-l", "--height", type=int, default=480, help="the frame height")
    ap.add_argument("-s", "--stream", type=int, default=1, help="stream to the dashboard")
    ap.add_argument("-a", "--address", default="192.168.24.25", help="address of the roborio")
    args = vars(ap.parse_args())

    # initialize some variables and frame holders
    width = args["width"]
    height = args["height"]
    frame   = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    grayed  = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    blurred = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    hsv     = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    mask    = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    # set up logging
    logging.basicConfig(level=logging.DEBUG)

    # connect to the roborio
    NetworkTables.initialize(server=args["address"])
    sd = NetworkTables.getTable("SmartDashboard")

    # set up the camera
    vs = VideoStream(src=0).start()

    # setup the stream if required
    if args["stream"] > 0:
        camServer = CameraServer.getInstance()
        framseStream = camServer.putVideo("Frame", width, height)
        maskStream = camServer.putVideo("Mask", width, height)

    while True:
        frame = vs.read()

        # get the bounds values from the dashboard
        lH = sd.getNumber("H-Lower", 0)
        lS = sd.getNumber("S-Lower", 0)
        lV = sd.getNumber("V-Lower", 0)

        uH = sd.getNumber("H-Upper", 180)
        uS = sd.getNumber("S-Upper", 255)
        uV = sd.getNumber("V-Upper", 255)

        # resize, blur, and convert to hsv space 
        frame = imutils.resize(frame, width=width, height=height)
        blurred = cv2.GaussianBlur(fram, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # create a mask, dilate and erode it
        mask = cv2.inRange(hsv, (lH, lS, lV), (uH, uS, uV))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find the contours in the mask
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # draw a center line on the frame
        cv2.line(frame, (320, 0), (320, height), (255, 255, 255), 1)

        # push the stream to the dashboard
        if args["stream"] > 0:
            frameStream.putFrame(frame)
            maskStream.putFrame(mask)

    # stop the camera, network, and fps counter
    vs.stop()
    NetworkTables.shutdown()

if __name__ == "__main__":
    main()
