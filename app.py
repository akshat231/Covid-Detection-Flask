import tensorflow as tf
model = tf.keras.models.load_model("best.h5")
from base64 import b64encode
import keras_ocr
from keras_ocr import tools
import cv2
import math
import io
import numpy as np
import os
import base64
from flask import Flask, render_template,request
app = Flask(__name__)
img_width=256; img_height=256
import cv2
from PIL import Image, ImageOps
import numpy as np
def findout(pred):
    return pred
def preprocess_image(im):
    im.resize((256, 256, 3), refcheck=False)
    a = np.array(im)
    a = np.expand_dims(a, axis = 0)
    a =np.divide(a,255)
    return a
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
def refine(prediction_groups,img):
    im=img
    for j in range(0,len(prediction_groups[0])):
        box = prediction_groups[0][j][1]
        x0, y0 = box[0]
        x1, y1 = box[1]
        x2, y2 = box[2]
        x3, y3 = box[3] 
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        g_mask = np.zeros(img.shape[:2],dtype='uint8')
        cv2.line(g_mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        cv2.line(g_mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,thickness)

        img = cv2.inpaint(img, g_mask, 7, cv2.INPAINT_NS)
        im=img
    return im

    # routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template('index.html')

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
     if request.method == 'POST':
        img = request.files['my_image']
        image = Image.open(img)
        image=cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB)
        pipeline = keras_ocr.pipeline.Pipeline()
        image = keras_ocr.tools.read(image)
        prediction_groups = pipeline.recognize([image])
        refined_image=refine(prediction_groups,image)
        resize_image=preprocess_image(refined_image)
        pred=np.argmax(model.predict(resize_image), axis=-1)
        prediction=findout(pred) 
        if prediction==0:
             s="It is a COVID!"
        elif prediction == 1:
             s="It is a Pneumonia!"
        elif prediction == 2:
             s="It is a Lung Opacity!"
        else:
             s="It is Normal!"
        return render_template("index.html", prediction = s)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
