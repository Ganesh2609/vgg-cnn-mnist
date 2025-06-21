import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import cv2 as cv  
import json  
import pandas as pd


def extract_feautures(directory):
    # Getting image paths
    data = []
    size = 224
    for root, subdir, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            img = cv.imread(path)
            img = cv.resize(img,(size,size))
            label = root.rsplit("\\",1)
            label = label[1].strip()
            data.append((img,label))
        
    # Extract Features
    model = VGG16()
    model = Model(inputs=model.inputs, outputs = model.layers[-2].output)
    features = []

    for dpt in data:
        img = dpt[0]
        arr = img_to_array(img)
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        pre = preprocess_input(arr)
        out = model.predict(pre, verbose=False)   # Verbose shows what happens inside the model
        features.append([out[0].tolist(),dpt[1]])
        
    final_features = {}    
    for i in features :
        key = i[1]
        value = final_features.get(key, ()) + (i[0],)
        final_features[key] = valueA
        
    return final_features


def signal_extraction(directory, number):
    files = list(os.walk(directory))[0][2]
    data = pd.DataFrame()
    
    for file in files:
        # Load data
        path = os.path.join(directory,file)
        current = pd.read_csv(path)
        data = pd.concat([data, current], ignore_index = True)
    
    # clean and group
    data_cleaned = data.dropna(how='any')
    grouped = data.groupby('type')
    selected_columns = list(range(2,34))
    groups = list(grouped.groups)
    groups.remove('Q')
    
    final_data = {}
    for group in groups:
        data = tuple(list(row) for row in grouped.get_group(group).iloc[number[0]:number[1], selected_columns].itertuples(index=False))
        final_data[group] = data
        
    return final_data
    
    
        