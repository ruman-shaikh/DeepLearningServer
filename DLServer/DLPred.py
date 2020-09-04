import os
import sys

from ModelList import PredToStr
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

def PreprocessImage(img, target_size):
	img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_CUBIC)
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = np.vstack([img])

	return img

def LoadModel(modelName):
	dirPath = os.path.join(os.getcwd(), "DLModels")
	filename = modelName + '.h5'
	filePath = os.path.join(dirPath, filename)
	if os.path.isfile(filePath) is False:
		raise FileNotFoundError("Model not found")

	model = load_model(filePath)
	target_size = model.layers[0].input_shape[1], model.layers[0].input_shape[2]
	if isinstance(target_size, tuple) is False or len(target_size) is not 2:
		raise TypeError("Input Dimension Mismatch")

	return model, target_size

def MakePrediction(modelName, img):
	code = 0
	message = ""
	response = {"code" : code, "message" : message}

	try:
		model, target_size = LoadModel(modelName)
	except FileNotFoundError:
		response["code"] = 1
		response["message"] = str(sys.exc_info()[1])
	except TypeError:
		response["code"] = 2
		response["message"]	= str(sys.exc_info()[1])
	else:
		img = PreprocessImage(img, target_size)
		results = model.predict(img)
		response["code"] = 3
		response["message"] = PredToStr(modelName, results) # classes[results[0][0]]

	return response