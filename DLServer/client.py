from __future__ import print_function
import requests
import json
import cv2

addr = 'http://192.168.0.114:5000'
test_url = addr + '/api/test'

# prepare headers for http request
ModelName = 'CatsVDogsKeras'
content_type = 'image/jpeg'
headers = {'content-type': content_type, 'modelname': ModelName}

img = cv2.imread('testcat1.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
print('Waiting for response')
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print(json.loads(response.text))
