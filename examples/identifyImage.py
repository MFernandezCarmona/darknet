#! /usr/bin/env python

import qi

import sys
import numpy as np
import cv2
import darknet as dn

from PyQt4.QtGui import QWidget, QImage, QApplication, QPainter
import vision_definitions
import argparse

from ctypes import *
from matplotlib import cm


def array_to_image(arr):
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr/255.0).flatten()
    data = dn.c_array(dn.c_float, arr)
    im = dn.IMAGE(w,h,c,data)
    return im

def detect2(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    num = c_int(0)
    pnum = pointer(num)
    dn.predict_image(net, im)
    dets = dn.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): dn.do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    #dn.free_image(im)
    #dn.free_detections(dets, num)
    return res



def getImage(video_service,imgClient):
        """
        Retrieve a new image from Nao.
        """
        _alImage = video_service.getImageRemote(imgClient)
        """
        Nao image format:
        [0]: width.
        [1]: height.
        [2]: number of layers.
        [3]: ColorSpace.
        [4]: time stamp from qi::Clock (seconds).
        [5]: time stamp from qi::Clock (microseconds).
        [6]: binary array of size height * width * nblayers containing image data.
        [7]: camera ID (kTop=0, kBottom=1).
        [8]: left angle (radian). 0.49
        [9]: topAngle (radian). 0.38
        [10]: rightAngle (radian). -0.49
        [11]: bottomAngle (radian). -0.38
        """
        return _alImage

def image_qi2np(dataImage):
    image = None
    if( dataImage != None ):
        image = np.reshape( np.frombuffer(dataImage[6], dtype='%iuint8' % dataImage[2]), (dataImage[1], dataImage[0], dataImage[2]))
        # image = np.fromstring(str(alImage[6]), dtype=np.uint8).reshape( alImage[1],alImage[0], dataImage[2])
    return image

def image_np2cv(npImage):
        # dirty way to use in cv2 or cv3
        if cv2.__version__ == '3.3.1-dev':
            open_cv_image = cv2.cvtColor(npImage, cv2.COLOR_BGR2RGB)
        else:
            open_cv_image = cv2.cvtColor(npImage, cv2.cv.CV_BGR2RGB)
        return open_cv_image


def main(session, robot_ip, port,camera,yoloCFG,yoloWeights,yoloData):
    
    video_service = session.service("ALVideoDevice")
    resolution = vision_definitions.kQVGA  # kQVGA =320 * 240  ,kVGA =640x480
    colorSpace = vision_definitions.kRGBColorSpace

    imgClient = video_service.subscribe("_clienteMe", resolution, colorSpace, 5)

    # Select camera.
    video_service.setParam(vision_definitions.kCameraSelectID, camera)
    
    # Darknet ... 
    net = dn.load_net(yoloCFG, yoloWeights , 0)
    meta = dn.load_meta(yoloData )

    #some constants for plotting
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:

        # get image
        result = getImage(video_service,imgClient)

        if result == None:
            print 'cannot capture.'
        elif result[6] == None:
            print 'no image data string.'
        else:

            np_img = image_qi2np(result)
            cv_img = image_np2cv(np_img)


            # send cv image to dn
            im = array_to_image(cv_img)
            dn.rgbgr_image(im)

            r = detect2(net, meta, im)
            print r
            print '..........................................'

            if r != []:
                cnt = 0
                while cnt < len(r):
                      name = r[cnt][0]
                      predict = r[cnt][1]
                      print (name+":"+str(predict))
                      x = r[cnt][2][0]
                      y = r[cnt][2][1]
                      w = r[cnt][2][2]
                      z = r[cnt][2][3]
                      #print (x, y, w, z)

                      x_max = int(round((2*x+w)/2))
                      x_min = int(round((2*x-w)/2))
                      y_min = int(round((2*y-z)/2))
                      y_max = int(round((2*y+z)/2))
                      #print (x_min, y_min, x_max, y_max)
                      pixel_list = [ x_min, y_min, x_max, y_max]
                      neg_index = [pixel_list.index(val) for val in pixel_list if val < 0]
                      object_color = cm.jet(cnt)[0:3]
                      cv2.rectangle(cv_img,(x_min,y_min),(x_max,y_max),(object_color), 2)
                      if neg_index == []:
                              cv2.rectangle(cv_img,(x_min,y_min-24), (x_min+10*len(name),y_min),object_color,-1)
                              cv2.putText(cv_img,name,(x_min,y_min-12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                      else:
                              if (y_min < 0 and x_min > 0):
                                      cv2.rectangle(cv_img,(x_min,0), (x_min+10*len(name),24),object_color,-1)
                                      cv2.putText(cv_img,name,(x_min,12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                              elif (x_min < 0 and y_min > 0):
                                      cv2.rectangle(cv_img,(0,y_min-24), (10*len(name),y_min),object_color,-1)
                                      cv2.putText(cv_img,name,(0,y_min-12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                              elif (x_min < 0 and y_min < 0):
                                      cv2.rectangle(cv_img,(0,0), (10*len(name),24),object_color,-1)
                                      cv2.putText(cv_img,name,(0,12), font, 0.5,(0,0,0),1,cv2.LINE_AA)
                      #cv2.imshow('image',cv_img)
                      #cropped = image.crop((x_min, y_min+20, x_max, y_max))
                      cnt+=1
                 
            # show image
            cv2.imshow("pepper-camera", cv_img)

        # exit by [ESC]
        if cv2.waitKey(33) == 27:
            break





if __name__ == "__main__":
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
    main(session, args.ip, args.port,args.camera,args.yoloCFG, args.yoloWeights, args.yoloData)