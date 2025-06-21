import import_external
import Feature
import json
import RKS

# training dataset

directory1 = 'F:/Ganesh/Amrita/Subjects/Sem 3/MFC -3/Final Project/Codes/Signal Application/Signal'
train_data = Feature.signal_extraction(directory1, (0,1150))
#train_data = RKS.map_rks(train_data, 200)
with open("train_signal_data.json", "w") as json_file:
    json.dump(train_data, json_file)

# testing dataset

directory2 = 'F:/Ganesh/Amrita/Subjects/Sem 3/MFC -3/Final Project/Codes/Signal Application/Signal'
test_data = Feature.signal_extraction(directory2, (1150,1250))
#test_data = RKS.map_rks(test_data, 200)
with open("test_signal_data.json", "w") as json_file:
    json.dump(test_data, json_file)
    