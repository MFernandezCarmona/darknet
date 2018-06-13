#! /usr/bin/env python

import qi

import sys

import argparse

import vision_definitions

import cv2

import time

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default="127.0.0.1",
                    help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
parser.add_argument("--port", type=int, default=9559,
                    help="Naoqi port number")
parser.add_argument("--camera", type=int, default=0,
                    help="CameraID: 0 up, 1 down, 2 depth")

parser.add_argument("--yoloCFG", type=str, default="/home/manolofc/workspace/darknet/cfg/yolov3.cfg",
                    help="Yolo cfg file with abslute path")
parser.add_argument("--yoloWeights", type=str, default="/home/manolofc/workspace/darknet/yolov3.weights",
                    help="Yolo weights file with abslute path")
parser.add_argument("--yoloData", type=str, default="/home/manolofc/workspace/darknet/cfg/coco-mfc.data",
                    help="Yolo data file with abslute path")
args = parser.parse_args()
session = qi.Session()
try:
    session.connect("tcp://" + args.ip + ":" + str(args.port))
except RuntimeError:
    print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
           "Please check your script arguments. Run with -h option for help.")
    sys.exit(1)


print ("Connecting to Darknet service")
DarknetSRV = session.service("DarknetSRV")
DarknetSRV.setNet(args.yoloCFG,args.yoloWeights)
DarknetSRV.setMeta(args.yoloData)



print ("Connecting to video service")
video_service = session.service("ALVideoDevice")
resolution = vision_definitions.kQVGA  # kQVGA =320 * 240  ,kVGA =640x480
colorSpace = vision_definitions.kRGBColorSpace

imgClient = video_service.subscribe("_cr71", resolution, colorSpace, 5)

# Select camera.
video_service.setParam(vision_definitions.kCameraSelectID, args.camera)


#some constants for plotting
font = cv2.FONT_HERSHEY_SIMPLEX

print ("Starting detection")
while True:
	result = video_service.getImageRemote(imgClient)
	#print(type(result[0]))
	r = DarknetSRV.identify(result)
	if r != []:
		cnt = 0
		while cnt < len(r):
			name = r[cnt][0]
			predict = r[cnt][1]
			print ("{0}: Confidence {1}".format(name,predict))

			x = r[cnt][2][0]
			y = r[cnt][2][1]
			w = r[cnt][2][2]
			z = r[cnt][2][3]
			print ("\t at [{0},{1}  {2},{3}]".format(x, y, w, z))
			print (".....................\n")
			cnt+=1
	print ("-------------------------\n\n")
	time.sleep(1)