#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import time
import seaborn as sns
from tensorflow.keras.losses import CategoricalCrossentropy


#%%
def DataToFlatten(data):
    num_bands = data[0,1].shape[-1]
    if data[0,1].shape[0] == 1:
        reshapeddata = np.empty((len(data), num_bands + 1), dtype = 'uint16')        
        reshapeddata[:,1:] = np.array([*data[:,1]]).reshape(len(data), num_bands)
        reshapeddata[:,0] = data[:,0]
    else:
        border_size = data[0,1].shape[0]**2
        reshapeddata = np.empty((len(data), border_size * num_bands + 1), dtype = 'uint16')
        reshapeddata[:,0] = data[:,0]
        for i in range(len(reshapeddata)):
            reshapeddata[i,1:] = data[i,1].flatten()
    return reshapeddata

def BestParEvaluator(parameters):
    rounded = np.round(parameters[:,0], 4)
    max_pos = np.where(rounded == max(rounded))[0]
    best = max_pos[0]
    return best



def DatasetBalance(data, numerosity = 3000, category_col = 0, dtype = 'uint16'):
    final_frame = []
    for category in np.unique(data[:,category_col]):
        positions = np.where(data[:,category_col] == category)[0]
        if len(positions) >= numerosity:
            chosen_index = np.random.choice(positions, size = numerosity, replace = False)
            final_frame.extend(data[chosen_index])
        else:
            augment_size = numerosity - len(positions)
            augment_index = np.random.choice(positions, size = augment_size, replace = True)
            rows_to_add = np.empty((augment_size, data.shape[-1]))
            for total in range(augment_size):
                noise = np.random.uniform(-4, 4, size = data.shape[-1] - 1)
                img = data[augment_index[total],category_col + 1:] + noise
                rows_to_add[total,:] = np.concatenate(([category], img))
                total += 1
            toappend = np.append(data[positions], rows_to_add, axis = 0)
            final_frame.extend(toappend)
    final_frame = np.array(final_frame, dtype = dtype)
    return final_frame


#%%


## LUCAS CATEGORIES DICTIONARY
import pandas as pd
# One digit:
code = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
description = ['Artificial land',
               'Cropland',
               'Woodland',
               'Shrubland',
               'Grassland',
               'Bareland',
               'Water',
               'Wetlands']
number = [1, 2, 3, 4, 5, 6, 7, 8]

onedigitdict = pd.DataFrame(np.transpose(np.array([code, description, number])), columns = ['code', 'description', 'number'])


#%%

# SPECIFY DATA PATH
path = 'C:/Users/drikb/Desktop/Tirocinio/EarthEngine/data/'


data = np.load(path + 'lucas_EU_3x3_12M_MEDIAN.npy',
               allow_pickle = True)

# we select central pixel of 3x3 images
data = np.array([[data[i,0], data[i,1][1,1,:].reshape((1,1,12))] for i in range(len(data))], dtype = 'object')

#%%

# define the digit of land cover we want to consider (advisable 1 digit)
digits = 1

data[:,0] = [data[i,0][digits-1] for i in range(len(data))]

data = data[np.where(data[:,0] != '8')]


#%%

for cat in range(len(code)):
    index = np.where(data[:,0] == code[cat])[0]
    data[index, 0] = number[cat]

#%%

np.random.shuffle(data)

reshapeddata = DataToFlatten(data)

lengths = []
for cat in np.unique(reshapeddata[:,0]):
    print(len(np.where(reshapeddata[:,0] == cat)[0]))
    lengths.append(len(np.where(reshapeddata[:,0] == cat)[0]))
    

#divide labels from inputs
df = reshapeddata[:,1:]
labels = reshapeddata[:,0]

# train, test, validation split
sample_size = min(lengths)
train_index = []
test_index = []
val_index = []
for cat in np.unique(labels):
    balindex = np.random.choice(np.where(labels == cat)[0], size = sample_size, replace = False)
    print(len(balindex))
    train_index.extend(balindex[0:int(sample_size*0.60)])
    test_index.extend(balindex[int(sample_size*0.60):int(sample_size*0.80)])
    val_index.extend(balindex[int(sample_size*0.80):-1])
    print(len(train_index), '\n',
          len(test_index), '\n',
          len(val_index), '\n',)

np.random.shuffle(train_index)
np.random.shuffle(val_index)
np.random.shuffle(test_index)


train_x, test_x, validation_x = df[train_index], df[test_index], df[val_index]
train_lab, test_lab, validation_lab = labels[train_index], labels[test_index], labels[val_index]


     
    
#%%

# MLP requires labels starting from 0, we train RF consistently

train_lab -= 1
test_lab -= 1
validation_lab -= 1

#a = parameters[np.where(parameters[:,0] == np.min(parameters[:,0]))[0][0], :]
#%%
### RF MODEL TUNING
num_classes = len(np.unique(train_lab))
s = time.time()
parameters = np.empty((0,3))
for regressors in range(int(train_x.shape[1]/2 + 5)): # for 1x1 we test even having all regressors
    regressors += 1
    for samples in [20, 50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]:
        print('Regressors: ', regressors, ', Samples: ', samples)
        RF_model = RandomForestClassifier(n_estimators = 100, # number of trees
                                          max_samples = samples, # observations per bootstrapped sample
                                          max_features = regressors, # number of regressors
                                          random_state = 42)
        RF_model.fit(train_x, train_lab)
        RF_pred = RF_model.predict(validation_x)
        valscore = metrics.accuracy_score(validation_lab, RF_pred)
        parameters = np.append(parameters, [valscore, int(regressors), int(samples)])
print(time.time()-s)

parameters = parameters.reshape((int(len(parameters)/3), 3))

#bestmodelpar = parameters[np.where(parameters[:,0] == np.max(parameters[:,0]))[0]]
bestmodelpar = parameters[BestParEvaluator(parameters),:]

### RF MODEL TESTING

bestmodel = RandomForestClassifier(n_estimators = 100, # number of trees
                                  max_samples = int(bestmodelpar[2]), # observations per bootstrapped sample
                                  max_features = int(bestmodelpar[1]), # number of regressors
                                  random_state = 42)

bestmodel.fit(train_x, train_lab)

best_pred = bestmodel.predict(test_x)


print("Accuracy = ", metrics.accuracy_score(test_lab, best_pred))
print('The theoretical accuracy of a random classifier is: ', 1/(len(np.unique(labels))))

matrix = confusion_matrix(test_lab, best_pred)
print(matrix.diagonal()/matrix.sum(axis=1))


print(classification_report(test_lab, best_pred))


sns.heatmap(matrix/1000, annot=True)
print(matrix/1000)


#%%

# use this chunk to save the model parameters and avoid tuning every time

bestmodel = RandomForestClassifier(n_estimators = 100, # number of trees
                                  max_samples = 2800, # observations per bootstrapped sample
                                  max_features = 9, # number of regressors
                                  random_state = 42)

bestmodel.fit(train_x, train_lab)

best_pred = bestmodel.predict(test_x)

from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_lab, best_pred))
print('The theoretical accuracy of a random classifier is: ', 1/(len(np.unique(labels))))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_lab, best_pred)
print(matrix.diagonal()/matrix.sum(axis=1))


from sklearn.metrics import classification_report
print(classification_report(test_lab, best_pred))


import seaborn as sns
sns.heatmap(matrix/1000, annot=True)
print(matrix/1000)


#%%

# predictors importance plot

import time

start_time = time.time()
importances = bestmodel.feature_importances_
std = np.std([tree.feature_importances_ for tree in bestmodel.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

forest_importances = pd.Series(importances, index=['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B1', 'B9', 'B11', 'B12'])

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()




    
    
