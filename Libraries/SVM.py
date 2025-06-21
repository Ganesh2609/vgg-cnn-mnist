import cvxpy as cp
import numpy as np
from itertools import combinations

# Binary SVM
def binary_svm(data_1, data_2, C):
    
    n = len(data_1[0])
    
    # Defining Variables
    w = cp.Variable(n)
    gam = cp.Variable()
    err = cp.Variable(len(data_1) + len(data_2))
    
    # Objective Function
    obj = cp.Minimize(0.5 * cp.norm(w, 'fro') + C*cp.sum(err))
    
    # Constraints
    cons = [err >= 0]
    for i in range(len(data_1)):
        cons.append((-(w @ data_1[i] - gam) + err[i]) >= 1)
    for i in range(len(data_2)):
        cons.append((w @ data_2[i] - gam + err[i+len(data_1)]) >= 1)
        
    # Solving the problem
    prb = cp.Problem(obj,cons)
    prb.solve(verbose=True)   # Verbose shows what happens inside the solver
    
    variables = {
        'w' : w.value.tolist(),
        'gam' : gam.value.tolist(),
        'err' : err.value.tolist()
    }
    
    return variables


# One vs Rest
def one_vs_rest_svm(data, C):
    n = len(data)
    output = []
    
    for i in range(n):
        label1 = list(data.keys())[i]
        data_2 = list(data[label1])
        
        data_1 = []
        for j in range(n):
            if i != j:
                label2 = list(data.keys())[j]
                data_1.extend(list(data[label2]))
         
        variables = binary_svm(data_1,data_2,C)     
        output.append((label1,variables))
    
    return output

# Testing one vs rest
def test_one_rest(classified, data):
    
    trained = {}
    for i in classified:
        trained[i[0]] = (i[1]['w'], i[1]['gam'])
    
    labels = list(trained.keys())
    distance = {i:0 for i in labels}

    for i in trained:
        w = trained[i][0]
        gam = trained[i][1]
        dist = ((np.array(w) @ data) - gam)/(np.linalg.norm(w))
        distance[i] = dist
        
    distance = dict(sorted(distance.items(), key = lambda x:x[1], reverse=True))
    return list(distance.keys())[0]


# One vs One
def one_vs_one_svm(data, C):
    
    keys = list(data.keys())
    output = []
    
    for i in combinations(keys,2):
        data_1 = data[i[0]]
        data_2 = data[i[1]]
        
        variables = binary_svm(data_1,data_2,C)      
        output.append((i,variables))

    return output

# Testing one vs one
def test_one_one(classified, data):
    
    trained = {}
    for i in classified:
        trained[tuple(i[0])] = (i[1]['w'], i[1]['gam'])
    
    labels = []
    for i in trained:
        labels.extend([i[0], i[1]])
    labels = set(labels)
    count = {i:0 for i in labels}     
    
    for label in trained:
        w = trained[label][0]
        gam = trained[label][1]
        
        if ((np.array(w) @ data) - gam) >= 0:
            count[label[1]] += 1
        else:
            count[label[0]] += 1
            
    count = dict(sorted(count.items(), key = lambda x:x[1], reverse =True))
    return list(count.keys())[0]
