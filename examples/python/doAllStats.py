
from scipy.misc import imread
import cv2
import sys, os
import glob
import darknet as dn
from ctypes import *
import random
from yaml import load
import re
import numpy as np
import json

def colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
	r += step
	g += step
	b += step
	r = int(r) % 256
	g = int(g) % 256
	b = int(b) % 256
	ret.append((b,g,r)) 
  return ret


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

def printObjestsInImage(r,cv_img):
	if r != []:
		cnt = 0
		numDetects = len(r) 
		colorList = colors(numDetects)
		#some constants for plotting
		font = cv2.FONT_HERSHEY_SIMPLEX
		while cnt < numDetects :
				name = r[cnt][0]
				predict = r[cnt][1]
				#print (name+": "+str(predict))

				x = r[cnt][2][0]
				y = r[cnt][2][1]
				w = r[cnt][2][2]
				z = r[cnt][2][3]

				x_max = int(round((2*x+w)/2))
				x_min = int(round((2*x-w)/2))
				y_min = int(round((2*y-z)/2))
				y_max = int(round((2*y+z)/2))

				pixel_list = [ x_min, y_min, x_max, y_max]
				neg_index = [pixel_list.index(val) for val in pixel_list if val < 0]
				object_color = colorList[cnt]

				name = name+" {:.2f}%".format(100*predict)
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
				cnt=cnt+1
	return cv_img.copy()

def array_to_image(arr):
	arr = arr.transpose(2,0,1)
	c = arr.shape[0]
	h = arr.shape[1]
	w = arr.shape[2]
	arr = (arr/255.0).flatten()
	data = dn.c_array(dn.c_float, arr)
	im = dn.IMAGE(w,h,c,data)
	return im


def showBasics(im,r):
		if r!=[]:
			cnt=0
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

			if False:
				if False:
					cropped = imcrop(arr,(x_min, y_min, x_max, y_max))
					cv2.imshow("pepper-camera", cropped)
				else:
					img2=printObjestsInImage(r,arr)
					cv2.imshow("pepper-camera", img2)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

			cv2.imwrite('pepper.png',cv_img)

			print r



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
	return res



def doStats(detectDict,avHist,numDect):
	res = []	
	color = ('b','g','r')
	for darknetN in detectDict:
		res.append(( darknetN, detectDict[darknetN]/numDect))
	res = sorted(res, key=lambda x: -x[1])
	
	avH={}
	for i,col in enumerate(color):
		 avH[col]= avHist[col].mean()/numDect

	return (res,avH)


#....................................

if False:
	yaml_file = './config.yaml'
	with open(yaml_file) as data_file:
		config = load(data_file)

	yoloCFG = config['yoloCFG']
	yoloWeights = config['yoloWeights']
	yoloData = config['yoloData']


yoloCFG="/home/manolofc/qi_ws/darknet/cfg/yolov3.cfg"
yoloWeights="/home/manolofc/qi_ws/darknet/yolov3.weights"
yoloData="/home/manolofc/qi_ws/darknet/cfg/coco-mfc.data"

#imgFolder = "/home/manolofc/Escritorio/objectVideos/testImages" 

# Darknet ... 
net = dn.load_net(yoloCFG, yoloWeights , 0)
meta = dn.load_meta(yoloData )


# all images here
imgFolderTotal = "/home/manolofc/Escritorio/objectVideos/caps"
imgListTotal=glob.glob(imgFolderTotal+'/*.jpg')

montrealNames=[]

# find out object names:
for file in imgListTotal:
	fileName=file.split('/')[-1]
	fileName=fileName.split('.')[0]
	montrealObj=re.sub(r"\d+", "", fileName)
	montrealNames.append(montrealObj)

uniqueNames=list(set(montrealNames))


data={}
filename='profit.txt'
for montrealObj in uniqueNames:
	imgList=glob.glob(imgFolderTotal+'/'+montrealObj+'*.jpg')

	#imgList=glob.glob(imgFolder+'/*.jpg')
	numFiles=len(imgList)
	
	color = ('b','g','r')

	#montrealObj='apple'

	detectedAs={}
	avHistr={}
	numDetects = 0

	print ("Robocup object: "+montrealObj)
	discards=['person','refrigerator','diningtable']

	for i in range(0,numFiles):
		imgFile = imgList[i]
		print "img: "+str(i+1)+" of "+str(numFiles)
		arr = cv2.imread(imgFile)
		
		if False:
			cv2.imshow("orig", arr)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		im = array_to_image(arr)
		dn.rgbgr_image(im)
		r = detect2(net, meta, im)

		#showBasics(im,r)
		if r==[]:
			print("Nothing...")
		else:
			discardImg = False
			
			validNames=[]
			for ri in r:
				if ri[0] not in discards:
					validNames.append(ri[0])
			
			if (len(validNames)==0):
				discardImg = True
				print("discarding empty or all invalid img")
			else:		
				img2=printObjestsInImage(r,arr)
				cv2.imshow("pepper-camera", img2)

			imgOk = False
			doExit = False

			while (not imgOk) and (not discardImg) and (not doExit):
				ch = cv2.waitKey(0)
				#if (ch!=-1):
				#	print ch
				imgOk = (ch == 32) # space
				discardImg = (ch == 27) # esc
				doExit = (ch == 120) # x
				if discardImg:
					print("Manual discard...")

			cv2.destroyAllWindows()

			if doExit:
				print("bye...")
				break 

			if imgOk:
				print("Ok...")

			cnt=0
			while (cnt < len(r)) and (imgOk):
				darknetName = r[cnt][0]
				if darknetName not in discards:
					predict = r[cnt][1]
					x = r[cnt][2][0]
					y = r[cnt][2][1]
					w = r[cnt][2][2]
					z = r[cnt][2][3]

					x_max = int(round((2*x+w)/2))
					x_min = int(round((2*x-w)/2))
					y_min = int(round((2*y-z)/2))
					y_max = int(round((2*y+z)/2))

					print ("Detected as: "+darknetName+":"+str(predict))

					cropped = imcrop(arr,(x_min, y_min, x_max, y_max))
					
					histr={}
		 			for i,col in enumerate(color):
		 				histr[col] = cv2.calcHist([cropped],[i],None,[256],[0,256])

					try:
						detectedAs[darknetName]+=predict
			 			for i,col in enumerate(color):
			 				avHistr[col]+= histr[col]

					except:
						detectedAs[darknetName]=predict
						for i,col in enumerate(color):
			 				avHistr[col]= histr[col]
					
					numDetects = numDetects + 1
				cnt=cnt+1

	# statistics
	(res,avHi) = doStats(detectedAs,avHistr,numDetects)
	print montrealObj
	print dict(res)
	print avHi

	data[montrealObj]=(dict(res),avHi)

with open(filename, 'wb') as outfile:
    json.dump(data, outfile)


montreal2Darknet={
'cereal':'None',
'bag':'None',
'cloth':'None',
'sponge':'None',
'apple':'apple',
'orange':'orange',
'cup':'orange',
'bowl':'orange',
'knife':'orange',
'fork':'orange',
'spoon':'orange',
'scrubby':'cake', #[('cake', 0.5717448790868124), ('cell phone', 0.21006598075230917)]
'noodles':'bottle', #[('bottle', 0.3586362500985463), ('book', 0.1409345688643279), ('cake', 0.13153265140674733), ('sandwich', 0.03073799389379996), ('pizza', 0.022061056560940213), ('fork', 0.016592724455727473), ('remote', 0.015102735272160283)]
#distinguish these!!!
'coke':'bottle', # 0.5838538010915121
'sprite':'bottle', # 0.57 
'chocolate drink':'bottle',
'grape juice':'cell phone', #[('cell phone', 0.4996661598032171), ('book', 0.17237312685359607)]
'orange juice':'cell phone',#[('cell phone', 0.4996661598032171), ('book', 0.17237312685359607)] ??"
'basket':'None',
'tray':'None',
'dish':'None',
'sausages':'bottle',
'paprika':'None',
'crackers':'book',
'chips':'book',
'pringles':'bottle'
}

"""
add counter
if nothing skip
if only person skip

"""