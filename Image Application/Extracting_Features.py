import import_external
import Feature
import json

# Testing dataset

test_dir = "F:/Ganesh/Amrita/Subjects/Sem 3/MFC -3/Final Project/Codes/Image Application/Handwritten Digits/test"
test_features = Feature.extract_feautures(test_dir) 

with open("test_digit_data.json", "w") as json_file:
    json.dump(test_features, json_file)

# Training dataset

train_dir = "F:/Ganesh/Amrita/Subjects/Sem 3/MFC -3/Final Project/Codes/Image Application/Handwritten Digits/train"
train_features = Feature.extract_feautures(train_dir) 

with open("train_digit_data.json", "w") as json_file:
    json.dump(train_features, json_file)
