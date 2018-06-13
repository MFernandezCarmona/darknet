#!/usr/bin/env python

'''
Performing visual flow on darknet bounding box
Based on opt_flow example from opencv
Like mfc2 but this one also plots upper part bbox 

'''


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import darknet as dn
from ctypes import *
from matplotlib import cm
import time
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


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))

    av_img_flow = (np.mean(np.mean(flow,axis=0),axis=0))
    #av_img_flow[0] = 10
    #av_img_flow[1]= 10
    x0 = w/2
    y0 = h/2
    #av_line = np.vstack([x0, y0, x0+av_img_flow[0], y0+av_img_flow[1]]).T.reshape(-1, 2, 2)
    #av_line = np.int32(av_line + 0.5)

    cv.line(vis, (x0, y0) , (x0+int(10*av_img_flow[0]), y0+int(10*av_img_flow[1])) , (255, 0, 0), 3)
    
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return (vis,av_img_flow)


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

def module(x,y):
    return np.sqrt((x*x)+(y*y))

def imcrop(img, bbox): 
    x1,y1,x2,y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)

    if img.ndim==3:
        return img[y1:y2, x1:x2, :]
    else:
        return img[y1:y2, x1:x2]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    if img.ndim==3:
        img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    else:
        img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
                  (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0))), mode="constant")

    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2


def biggestTuple(tA,tB):
    (Ax_min, Ay_min, Ax_max, Ay_max )= tA
    (Bx_min, By_min, Bx_max, By_max )= tB

    tC = ( min(Ax_min,Bx_min) , min(Ay_min,By_min), max(Ax_max,Bx_max), max(Ay_max,By_max))
    #print(tA)
    #print(tB)
    #print(tC)
    return tC

def detectOnImage(img,net,meta):
    cv_img = img.copy()
    im = array_to_image(cv_img)
    dn.rgbgr_image(im)

    r = detect2(net, meta, im)
    return r,cv_img

def drawConfidence(img,prob,bbox):
    (x_min,y_min,x_max,y_max)= bbox
    pixel_list = [ x_min, y_min, x_max, y_max]
    neg_index = [pixel_list.index(val) for val in pixel_list if val < 0]

    object_color = (0,255,0)
    label = "{:.2%}".format(prob)
    font = cv.FONT_HERSHEY_SIMPLEX


    cv.rectangle(img,(x_min,y_min),(x_max,y_max),(object_color), 2)
    if neg_index == []:
          cv.rectangle(img,(x_min,y_min-24), (x_min+10* len(label),y_min),object_color,-1)
          cv.putText(img,label,(x_min,y_min-12), font, 0.5,(0,0,0),1,cv.LINE_AA)
    else:
          if (y_min < 0 and x_min > 0):
                  cv.rectangle(img,(x_min,0), (x_min+10*len(label),24),object_color,-1)
                  cv.putText(img,label,(x_min,12), font, 0.5,(0,0,0),1,cv.LINE_AA)
          elif (x_min < 0 and y_min > 0):
                  cv.rectangle(img,(0,y_min-24), (10*len(label),y_min),object_color,-1)
                  cv.putText(img,label,(0,y_min-12), font, 0.5,(0,0,0),1,cv.LINE_AA)
          elif (x_min < 0 and y_min < 0):
                  cv.rectangle(img,(0,0), (10*len(label),24),object_color,-1)
                  cv.putText(img,label,(0,12), font, 0.5,(0,0,0),1,cv.LINE_AA)

def drawBB(img,bbox):
    (x_min,y_min,x_max,y_max)= bbox
    pixel_list = [ x_min, y_min, x_max, y_max]
    neg_index = [pixel_list.index(val) for val in pixel_list if val < 0]
    object_color = (255,255,0)
    

    cv.rectangle(img,(x_min,y_min),(x_max,y_max),(object_color), 2)
    


def getPixelT(ri):
    x = ri[2][0]
    y = ri[2][1]
    w = ri[2][2]
    z = ri[2][3]
    #print (x, y, w, z)

    x_max = int(round((2*x+w)/2))
    x_min = int(round((2*x-w)/2))
    y_min = int(round((2*y-z)/2))
    y_max = int(round((2*y+z)/2))

    #print (x_min, y_min, x_max, y_max)
    pixel_list = [ x_min, y_min, x_max, y_max]
    pixel_tuple = (x_min, y_min, x_max, y_max)
    return (pixel_list, pixel_tuple)

if __name__ == '__main__':
        import sys
        print(__doc__)
        
        try:
            fn = sys.argv[1]
        except IndexError:
            fn = 0


        cam = cv.VideoCapture(fn)
        ret, prev = cam.read()

        prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        prev_pixel_tuple = (0,0,0,0)
        firstTime = True

        show_hsv = False
        show_glitch = False
        cur_glitch = prev.copy()
        count = 0

        #Darknet ... 
        yoloCFG = "/home/manolofc/workspace/darknet/cfg/yolov3.cfg"
        yoloWeights = "/home/manolofc/workspace/darknet/yolov3.weights"
        yoloData = "/home/manolofc/workspace/darknet/cfg/coco-mfc.data"
        net = dn.load_net(yoloCFG, yoloWeights , 0)
        meta = dn.load_meta(yoloData )

        #some constants for plotting
        font = cv.FONT_HERSHEY_SIMPLEX


        while True:
                #get the image
                ret, img0 = cam.read()
                #time.sleep(0.2)
                #get the image
                ret, img1 = cam.read()

                r0,cv_img0 = detectOnImage(img0,net,meta)
                r1,cv_img1 = detectOnImage(img1,net,meta)

                global_img = cv_img1.copy()

                humanProb = 0
                if (r0 != []) and (r1 != []):
                    cnt = 0
                    while cnt < (min(len(r0),len(r1))):
                            name0 = r0[cnt][0]
                            predict0 = r0[cnt][1]

                            name1 = r1[cnt][0]
                            predict1 = r1[cnt][1]


                            print (name0+": "+str(predict0))
                            print (name1+": "+str(predict1))


                            a = ('person' in name0) 
                            a = a and ('person' in name1)
                            if  a :
                                    humanProb = predict1

                                    (pixel_list0, pixel_tuple0) = getPixelT(r0[cnt])
                                    (pixel_list1, pixel_tuple1) = getPixelT(r1[cnt])

                                    showCropped = True
#                                    if showCropped:
#                                        cropped = imcrop(cv_img0, pixel_tuple0) 
#                                        cv.imshow('ROI',cropped)
#                                    else:
#                                        cv.imshow('ROI',cv_img0)
                                    
                                    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
                                    gray0 = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)

                                    max_tuple = biggestTuple(pixel_tuple1,pixel_tuple0)

                                    gray_cropped0 = imcrop(gray0,max_tuple)
                                    gray_cropped1 = imcrop(gray1,max_tuple)
                                    
                                    flow = cv.calcOpticalFlowFarneback(gray_cropped1, gray_cropped0, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                                    h, w =  gray_cropped1.shape[:2]
                                    factor = 1/3.0
                                    hi = int(h*factor)
                                    (upper_flow_img,upper_v) = draw_flow(gray_cropped1[0:hi,:,], flow[0:hi,:,])
                                    (lower_flow_img,lower_v) = draw_flow(gray_cropped1[hi:-1,:,], flow[hi:-1,:,])

                                    #probability of a wave=upper_half/(upper + lower)
                                    global_flow = module((upper_v[0]+lower_v[0])/2.0,(upper_v[1]+lower_v[1])/2)
                                    up_flow = module(upper_v[0],upper_v[1])  
                                    down_flow = module(lower_v[0],lower_v[1])

                                    if global_flow>0.4:
                                        waveProb =  up_flow/(up_flow+down_flow) 
                                    else:
                                        waveProb = 0

                                    # total prob=a*b
                                    prob = humanProb*waveProb
                                    print("\n********************")
                                    print("final waving Prob: "+str(prob))
                                    print("********************\n")

                                    drawConfidence(global_img,prob,max_tuple)
                                    (x_m, y_m, x_M, y_M) = max_tuple
                                    k =  (y_M - y_m)/2.0

                                    bbox = (x_m, y_m, x_M, int(y_M-k)) 
                                    drawBB(global_img,bbox)
#                                    if showCropped:
#                                        vis = np.concatenate((upper_flow_img, lower_flow_img), axis=0)
#                                        cv.imshow('flow', vis)
                                    
                            cnt+=1
                            if showCropped:
                                cv.imshow('Detection', global_img)

                ch = cv.waitKey(5)
                if ch == 27:
                    break
        cv.destroyAllWindows()
