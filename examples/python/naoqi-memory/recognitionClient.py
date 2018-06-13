#! /usr/bin/env 

from naoqi import ALProxy
import argparse




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--ip", type=str, default="127.0.0.1",
		help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
	parser.add_argument("--port", type=int, default=9559,
		help="Naoqi port number")
	parser.add_argument("--objectSRV", type=str, default="detectedObjects",
		help="recognized objets service. default detectedObjects")

	args = parser.parse_args()
	try:
		# create proxy on ALMemory
		memProxy = ALProxy("ALMemory",args.ip,args.port)

		#insertData. Value can be int, float, list, string

		#getData
		while True:
			detection=memProxy.getData(args.objectSRV)
			print "We have detected  "+ str(len(detection))+" objects"
			for detect in detection:
				print "- "+ detect[0]
			print "............................................... "

	except RuntimeError,e:
		# catch exception
		print "error insert data", e