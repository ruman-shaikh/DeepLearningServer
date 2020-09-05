import numpy as np

def ImageNetCatsVDogs(results):
	classes = {0.0 : "Cat", 1.0 : "Dog"}
	return classes[results[0][0]]

def CatsVDogs(results):
	classes = {0.0 : "Cat", 1.0 : "Dog"}
	return classes[results[0][0]]

def CatsVDogsKeras(results):    
    classes = {0.0 : "Cat", 1.0 : "Dog"}
    return classes[np.argmax(results)]

def PredToStr(ModelName, results):
	ModelList = {
		"ImageNetCatsVDogs" : ImageNetCatsVDogs,
		"CatsVDogs" : CatsVDogs,
		"CatsVDogsKeras" : CatsVDogsKeras
	}
	return ModelList[ModelName](results)