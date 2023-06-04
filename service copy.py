import numpy as np
from config import IMG_DIR
from tensorflow.keras.models import Model
import shutil, pickle, logging, tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from utils import generate_random_string, get_matching_strings
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


def save_image(image):
    """Save Image to local storage temporary"""

    image.filename = IMG_DIR + generate_random_string()
    with open(image.filename, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
    return image.filename

def process_image(image_path):
    """Process the image"""

    image = load_img(image_path, target_size = (224, 224))
    image_arr = img_to_array(image)
    image_arr = image_arr.reshape((1, image_arr.shape[0], image_arr.shape[1], image_arr.shape[2]))
    image = preprocess_input(image_arr)
    return image

def feature_training(image, image_path, features):
    """Training the features on VGG19 model"""

    image_id = image_path.split(".")[0]
    model = VGG19()
    model = Model(inputs = model.inputs, outputs=model.layers[-2].output)
    feature = model.predict(image, verbose = 0)
    features[image_id] =  feature
    return feature


def idx_to_word(integer, tokenizer):
    """Convert the tokens"""

    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    """Predict the captions"""

    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        
        yhat = model.predict([image, sequence], verbose = 0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if not word:
            break

        in_text += " " + word
        if word == "endseq": break
    return in_text

def process_list_image(img_list):
    idx = min(10, len(img_list))
    ans = []
    for i in range(idx):
        ans.append(IMG_DIR + img_list[i].split(".")[0] + ".jpg")
    return ans

def predict_result(image):
    """Club all the functions into single one and converted to get tle list of images"""

    image_path =  save_image(image)
    image_name = image_path.split("/")[-1]
    image = process_image(image_path)

    max_length = 35
    with open("all_captions.pkl", "rb") as file:
        all_captions = pickle.load(file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)

    with open("features.pkl", "rb") as file:
        features = pickle.load(file)
    model = load_model('model.h5')

    mapping = pickle.load(open("mapping.pkl", "rb"))

    feature = feature_training(image, image_path, features)
    features[image_name] = feature


    predicted_caption = predict_caption(model, feature, tokenizer, max_length = max_length)
    result = get_matching_strings(main_string = predicted_caption, min_keyword_matches=int(len(predicted_caption.split())*.9))
    result = process_list_image(result)
    mapping[image_name] = (predicted_caption + " ")*10
    all_captions.append(predicted_caption)

    with open("features.pkl", "wb") as file:
        pickle.dump(features, file)
    with open("mapping.pkl", "wb") as file:
        pickle.dump(mapping, file)
    with open("all_captions.pkl", "wb") as file:
        pickle.dump(all_captions, file)
    
    return {"success" : True, "payload": result, "captions" : predicted_caption, "file name" : image_name}