from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

class_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
               'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 'Shepherds Purse',
               'Small-flowered Cranesbill', 'Sugar beet']


app = FastAPI()
model = load_model("plant_seedlings_model.h5")

def read_image(file) -> np.array:
    img = image.load_img(file, target_size=(224, 224))
    expanded_array = image.img_to_array(img)
    normalized_array = np.expand_dims(expanded_array, axis=0) /225
    return normalized_array