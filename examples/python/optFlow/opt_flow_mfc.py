#!/usr/bin/env python

'''
Performing visual flow on darknet bounding box
Based on opt_flow example from opencv
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import darknet as dn
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

    cv.line(vis, (x0, y0) , (x0+int(10*av_img_flow[0]), y0+int(10*av_img_flow[1])) , (0, 0, 255), 3)
    
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
        yoloCFG = "/home/manolofc/qi_ws/darknet/cfg/yolov3.cfg"
        yoloWeights = "/home/manolofc/qi_ws/darknet/yolov3.weights"
        yoloData = "/home/manolofc/qi_ws/darknet/cfg/coco-mfc.data"
        net = dn.load_net(yoloCFG, yoloWeights , 0)
        meta = dn.load_meta(yoloData )

        #some constants for plotting
        font = cv.FONT_HERSHEY_SIMPLEX


        while True:
                #get the image
                ret, img0 = cam.read()

                #do we hav human? crop image to human
                # send cv image to dn
                cv_img = img0.copy()
                im = array_to_image(cv_img)
                dn.rgbgr_image(im)

                r = detect2(net, meta, im)
                humanProb = 0
                if r != []:
                    cnt = 0
                    while cnt < len(r):
                            name = r[cnt][0]
                            predict = r[cnt][1]
                            #print (name+": "+str(predict))

                            if  'person' in name:
                                    humanProb = predict

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
                                    pixel_tuple = (x_min, y_min, x_max, y_max)
                                    #print(pixel_tuple)
                                    neg_index = [pixel_list.index(val) for val in pixel_list if val < 0]
                                    #object_color = cm.jet(5*cnt)[0:3]
                                    object_color = (255,0,0)

                                    cv.rectangle(cv_img,(x_min,y_min),(x_max,y_max),(object_color), 2)
                                    if neg_index == []:
                                        cv.rectangle(cv_img,(x_min,y_min-24), (x_min+10*len(name),y_min),object_color,-1)
                                        cv.putText(cv_img,name,(x_min,y_min-12), font, 0.5,(0,0,0),1,cv.LINE_AA)
                                    else:
                                        if (y_min < 0 and x_min > 0):
                                            cv.rectangle(cv_img,(x_min,0), (x_min+10*len(name),24),object_color,-1)
                                            cv.putText(cv_img,name,(x_min,12), font, 0.5,(0,0,0),1,cv.LINE_AA)
                                        elif (x_min < 0 and y_min > 0):
                                            cv.rectangle(cv_img,(0,y_min-24), (10*len(name),y_min),object_color,-1)
                                            cv.putText(cv_img,name,(0,y_min-12), font, 0.5,(0,0,0),1,cv.LINE_AA)
                                        elif (x_min < 0 and y_min < 0):
                                            cv.rectangle(cv_img,(0,0), (10*len(name),24),object_color,-1)
                                            cv.putText(cv_img,name,(0,12), font, 0.5,(0,0,0),1,cv.LINE_AA)
    
                                    #cropped = image.crop((x_min, y_min+20, x_max, y_max))
                                    showCropped = True
                                    if showCropped:
                                        cropped = imcrop(cv_img, pixel_tuple) 
                                        cv.imshow('ROI',cropped)
                                    else:
                                        cv.imshow('ROI',cv_img)
                                    img = img0
                                    #img = cropped
                                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                                    if firstTime:
                                        prevgray = gray
                                        prev_pixel_tuple = pixel_tuple
                                        firstTime = False


                                    max_tuple = biggestTuple(pixel_tuple,prev_pixel_tuple)

                                    gray_cropped = (imcrop(gray,max_tuple))
                                    flow = cv.calcOpticalFlowFarneback(gray_cropped, imcrop(prevgray,max_tuple), None, 0.5, 3, 15, 3, 5, 1.2, 0)

                                    prevgray = gray
                                    prev_pixel_tuple = pixel_tuple

                                    h, w =  gray_cropped.shape[:2]
                                    factor = 1/3.0
                                    hi = int(h*factor)
                                    (upper_flow_img,upper_v) = draw_flow(gray_cropped[0:hi,:,], flow[0:hi,:,])
                                    (lower_flow_img,lower_v) = draw_flow(gray_cropped[hi:-1,:,], flow[hi:-1,:,])

                                    #probability of a wave=upper_half/(upper + lower)
                                    global_flow = module((upper_v[0]+lower_v[0])/2.0,(upper_v[1]+lower_v[1])/2)
                                    up_flow = module(upper_v[0],upper_v[1])  
                                    down_flow = module(lower_v[0],lower_v[1])

                                    #print("global: "+str(global_flow))
                                    #print("up: "+str(up_flow))
                                    #print("down: "+str(down_flow))

                                    if global_flow>0.4:
                                        waveProb =  up_flow/(up_flow+down_flow) 
                                    else:
                                        waveProb = 0

                                    # total prob=a*b
                                    prob = humanProb*waveProb
                                    if count == 5:
                                        print("\n********************")
                                        print("final waving Prob: "+str(prob))
                                        print("********************\n")
                                        count = 0
                                    else:
                                        count = count+1
                                    if showCropped:
                                        #cv.imshow('upper_flow', upper_flow_img)
                                        #cv.imshow('lower_flow', lower_flow_img)
                                        vis = np.concatenate((upper_flow_img, lower_flow_img), axis=0)
                                        cv.imshow('flow', vis)

                                    if show_hsv:
                                        cv.imshow('flow HSV', draw_hsv(flow))
                                    if show_glitch:
                                        cur_glitch = warp_flow(cur_glitch, flow)
                                        cv.imshow('glitch', cur_glitch)
                            cnt+=1

                ch = cv.waitKey(5)
                if ch == 27:
                    break
                if ch == ord('1'):
                    show_hsv = not show_hsv
                    print('HSV flow visualization is', ['off', 'on'][show_hsv])
                if ch == ord('2'):
                    show_glitch = not show_glitch
                if show_glitch:
                    cur_glitch = img.copy()
                    print('glitch is', ['off', 'on'][show_glitch])
        cv.destroyAllWindows()
