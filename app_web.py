from flask import Flask, render_template, request, redirect, url_for, send_from_directory


from PIL import Image

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import *
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
import os.path

app = Flask(__name__)

#Articles = Articles()

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


# @app.route('/articles')
# def articles():
#     return render_template('articles.html', articles=Articles)

@app.route('/validation/apple/<path:path>')
def send_js1(path):
    return send_from_directory('validation/apple/', path)

@app.route('/validation/banana/<path:path>')
def send_js2(path):
    return send_from_directory('validation/banana/', path)

@app.route('/validation/beetroot/<path:path>')
def send_js3(path):
    return send_from_directory('validation/beetroot/', path)

@app.route('/validation/bell_pepper/<path:path>')
def send_js4(path):
    return send_from_directory('validation/bell_pepper/', path)

@app.route('/validation/cabbage/<path:path>')
def send_js5(path):
    return send_from_directory('validation/cabbage/', path)

@app.route('/validation/capsicum/<path:path>')
def send_js6(path):
    return send_from_directory('validation/capsicum/', path)

@app.route('/validation/carrot/<path:path>')
def send_js7(path):
    return send_from_directory('validation/carrot/', path)

@app.route('/validation/cauliflower/<path:path>')
def send_js8(path):
    return send_from_directory('validation/cauliflower/', path)

# @app.route('/article/<string:id>')
# def article(id):
#     return render_template('article.html', articles=Articles, id=id)


def process_img(path):
    img = Image.open(path)
    if img.size[0] > img.size[1]:
        scale = 100 / img.size[1]
        new_h = int(img.size[1]*scale)
        new_w = int(img.size[0]*scale)
        new_size = (new_w, new_h)
    else:
        scale = 100 / img.size[0]
        new_h = int(img.size[1]*scale)
        new_w = int(img.size[0]*scale)
        new_size = (new_w, new_h)

    resized = img.resize(new_size)
    resized_img = np.array(resized, dtype=np.uint8)

    left = 0
    right = left + 100
    up = 0
    down = up + 100
    cropped = resized.crop((left, up, right, down))
    cropped_img = np.array(cropped, dtype=np.uint8)

    cropped_img = cropped_img / 255

    return cropped_img

model = load_model('96acc_model.h5')

@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'test_images/')
    # print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        # print(file)
        filename = file.filename
        destination = target+filename
        # print(destination)
        file.save(destination)	

    newDes = os.path.join('test_images/'+filename)

    print(newDes)
    
    train_categories = []
    for index, i in enumerate(os.listdir("./data/archive/train")):
        train_categories.append(i)
    print(train_categories)
    



    
    cropped_img = process_img(newDes)
    X = np.reshape(cropped_img, newshape=(1, cropped_img.shape[0], cropped_img.shape[1], cropped_img.shape[2]))
    # print("this is hereeeeee" + str(X))
    
    prediction_multi = model.predict(x=X)

    # print("this is here" )

    fruit_name = train_categories[np.argmax(prediction_multi)]
    print(fruit_name)
    # print("we here")
    results = []
    for index, i in enumerate(os.listdir("./validation/"+fruit_name)):
        results.append(i)

    print(results)

    result = [fruit_name,results]

    return render_template('about.html',results = result)
    #return (results, destination)




if __name__ == '__main__':
    app.run(debug=True)
    app.run(port=5000, debug=True, use_reloader=False)
