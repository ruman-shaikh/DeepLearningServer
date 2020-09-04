from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

from DLPred import MakePrediction

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # perform prediction
    response = MakePrediction('ImageNetCatsVDogs', img)
    # convert to JSON
    response_pickled = jsonpickle.encode(response)
    # send the response
    return Response(response=response_pickled, status=200, mimetype="application/json")

# start flask app
app.run(host="0.0.0.0", port=5000)