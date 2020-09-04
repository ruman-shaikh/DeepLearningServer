# DeepLearningServer

Modules required for this project
1. opencv-python
2. tensorflow-gpu
3. numpy
4. matplotlib
5. flask

The file server.py is the flask web server.
The file client.py is the python client designed to interact with the above mentioned flask web server.

For demo a mobilenet pre-trained model is fine-tuned to classify images of cats and dogs.
This model is used as an example but the server should be able to work with any tensorflow model.
The creation of the model and everthing related with it can be found in the mobnet.py file.
After training the model is saved as ImageNetCatsVDogs.h5 in the DLModels folder.

The DLPred.py file contains all the function required to preprocess the image, load the model and make the prediction.
The ModelList.py file contains the list of saved models and the function required to convert a tensorlfow model prediction output to a string of the predicted class.
