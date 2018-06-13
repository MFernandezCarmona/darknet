"""
Darknet as a NAOqi service
Run with python darknetsrv.py --qi-url 10.5.34.59
"""

__version__ = "0.0.1"

__copyright__ = "OPENSOURCE"
__author__ = 'mfernandezcarmona'
__email__ = 'mfernandezcarmona@lincoln.ac.uk'


import qi

import numpy as np
import cv2
import darknet as dn
from ctypes import *

from yaml import load
import stk.runner
import stk.events
import stk.services
import stk.logging

class DarknetSRV(object):
    
    cocoCategories = {'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
                  'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
                  'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
                  'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
                  'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                  'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed',
                  'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave',
                  'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'}



    "NAOqi service example (set/get on a simple value)."
    APP_ID = "com.aldebaran.DarknetSRV"
    def __init__(self, qiapp):
        # generic activity boilerplate
        self.qiapp = qiapp
        self.events = stk.events.EventHelper(qiapp.session)
        self.s = stk.services.ServiceCache(qiapp.session)
        self.logger = stk.logging.get_logger(qiapp.session, self.APP_ID)
        # Internal variables
        self.net = None
        self.meta = None
        self.isReady = False

        yaml_file = './config.yaml'
        with open(yaml_file) as data_file:
               config = load(data_file)

        yoloCFG = config['yoloCFG']
        yoloWeights = config['yoloWeights']
        yoloData = config['yoloData']

        #yoloCFG="/home/manolofc/workspace/darknet/cfg/yolov3.cfg"
        #yoloWeights"/home/manolofc/workspace/darknet/yolov3.weights"
        #yoloData="/home/manolofc/workspace/darknet/cfg/coco-mfc.data"

        #print("You still need to add weights, config and meta")
        self.setNet(yoloCFG,yoloWeights)
        self.setMeta(yoloData)
        print("Darknet service loaded")

    @qi.bind(returnType=qi.AnyArguments, paramsType=[qi.List(qi.UInt16)]) # == python long??
    def identify(self, img):
        "Get image return identification"
        #print("Received image")
        r = []
        if self.isReady:
            if img == None:
                print 'Error: invalid capture.'
            elif img[6] == None:
                print 'Error: no image data string.'
            else:
                #print("Image is correct")
                np_img = self.image_qi2np(img)
                cv_img = self.image_np2cv(np_img)


                # send cv image to dn
                im = self.array_to_image(cv_img)
                dn.rgbgr_image(im)
                #print("Detecting")
                r = self.detect2(self.net, self.meta, im)
                #print("Detection done")
        return r


    @qi.bind(returnType=qi.Void, paramsType=[qi.String, qi.String])
    def setNet(self, yoloCFG,yoloWeights):
        "Set Net"

        if (self.net==None):
            self.net = dn.load_net(yoloCFG, yoloWeights , 0)
            print("cfg and weights added")

        self.isReady = (self.net!=None) and (self.meta!=None)
        

    @qi.bind(returnType=qi.Void, paramsType=[qi.String])
    def setMeta(self, yoloData):
        "Set meta"
        if (self.meta==None):
            print("Using yoloData file:"+yoloData)
            self.meta = dn.load_meta(yoloData)
            print("meta added")

        self.isReady = (self.net!=None) and (self.meta!=None)
        

# Private? methods
    def image_qi2np(self,dataImage):
        image = None
        if( dataImage != None ):
            image = np.reshape( np.frombuffer(dataImage[6], dtype='%iuint8' % dataImage[2]), (dataImage[1], dataImage[0], dataImage[2]))
            # image = np.fromstring(str(alImage[6]), dtype=np.uint8).reshape( alImage[1],alImage[0], dataImage[2])
        return image

    def image_np2cv(self,npImage):
            # dirty way to use in cv2 or cv3
            if cv2.__version__ == '3.3.1-dev':
                open_cv_image = cv2.cvtColor(npImage, cv2.COLOR_BGR2RGB)
            else:
                open_cv_image = cv2.cvtColor(npImage, cv2.cv.CV_BGR2RGB)
            return open_cv_image


    def detect2(self,net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
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
        return res

    def array_to_image(self,arr):
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = (arr/255.0).flatten()
        data = dn.c_array(dn.c_float, arr)
        im = dn.IMAGE(w,h,c,data)
        return im

####################
# Setup and Run
####################

if __name__ == "__main__":
    stk.runner.run_service(DarknetSRV)

