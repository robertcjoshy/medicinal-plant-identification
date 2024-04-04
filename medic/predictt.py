import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
def predict_class(image_file):
    img = image.load_img(image_file,target_size=(224,224))
    print(type(img))
    img = np.asarray(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    saved_model = load_model("models/vgg16_v3.h5")
    pred_array = saved_model.predict(img)
    pred_class = pred_array.argmax(axis=-1)[0]
    print(pred_class)
    print("-----------------------------------------")
    return pred_class,pred_array[0][pred_class]

