import import_external
import SVM
import json
import numpy as np

with open("one_vs_one_trained_set.json", "r") as json_file:
    train_data = json.load(json_file)

with open("test_digit_data.json", "r") as json_file:
    test_data = json.load(json_file)

sno = 1
avg = []
print("Accuracy values : ")
for i in test_data:
    data_points = test_data[i]
    count = 0
    max_val = len(data_points)
    
    for j in data_points:
        label = SVM.test_one_one(train_data, j)
        if label == i:
            count+=1
        
    percentage = (count*100)/max_val
    avg.append(percentage)
    print(sno,") Class ",i," = ",percentage,sep="")
    sno+=1
    
print("Average accuracy : ",sum(avg)/len(avg))