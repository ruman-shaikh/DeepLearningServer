def ImageNetCatsVDogs(results):
	classes = {0.0 : "Cat", 1.0 : "Dog"}
	return classes[results[0][0]]

def PredToStr(ModelName, results):
	ModelList = {
		"ImageNetCatsVDogs" : ImageNetCatsVDogs
	}
	return ModelList[ModelName](results)
