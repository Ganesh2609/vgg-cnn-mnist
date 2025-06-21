import import_external
import SVM
import json
import numpy as np

with open("train_signal_data.json", "r") as json_file:
    data = json.load(json_file)

trained_set = SVM.one_vs_rest_svm(data, 1000)

with open("one_vs_rest_trained_set.json", "w") as json_file:
    json.dump(trained_set,json_file)
    