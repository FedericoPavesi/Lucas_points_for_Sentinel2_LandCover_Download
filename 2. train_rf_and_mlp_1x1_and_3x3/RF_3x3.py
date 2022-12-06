#%%
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import time
import seaborn as sns

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


def RotationAugmentation(data, categories, length = 2000):
    augment_list = []
    for cat in categories:
        index = np.where(data[:,0] == cat)[0]
        to_augment = length - len(index)
        print('We need to add ', to_augment, 'elements')
        if to_augment <= 0:
            print('Category already enough represented, skipping...')
            continue
        
        if to_augment/len(index) < 0:
            chosen = np.random.choice(index, size = to_augment, replace = False)
            print('Chosen done...')
            for i in chosen:
                rot_img = rotate(data[i,1], angle = 180, preserve_range = True).astype('uint16')
                augment_list.append([cat, rot_img])
        else:
            augment_number = int(to_augment/len(index)) + 1
            if augment_number > 3:
                print('For category ', cat, ' we are not able to reach the objective \n', 'The maximum reachable number is ', 3*len(index))
                to_augment = 3*len(index)
                augment_number = 3
            for i in index:
                for shift_num in range(1, augment_number+1):
                    rot_img = rotate(data[i,1], angle = 90*shift_num, preserve_range = True).astype('uint16')
                    augment_list.append([cat, rot_img])
    
    augment_frame = np.array(augment_list, dtype = 'object')
    return augment_frame

def OrderBands(element, from_order = ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9'], to_order = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B1', 'B9', 'B11', 'B12']):
    from_order = np.array(from_order)
    to_order = np.array(to_order)
    image = element[1]
    newimage = np.empty((image.shape[0], image.shape[1], image.shape[2]), dtype = 'uint16')
    for band_pos in range(len(to_order)):
        band = np.where(from_order == to_order[band_pos])[0]
        newimage[band_pos,:,:] = image[band,:,:]
        
    return np.array([element[0], newimage], dtype = 'object')

def Reshape3x3(element):
    image = element[1]
    
    newshape = np.empty((3,3,12), dtype = 'uint16')
    
    for band in range(image.shape[0]):
        newshape[:,:,band] = image[band,:,:]
        
    newelement = np.empty((2), dtype = 'object')
    
    newelement[0] = element[0]
    newelement[1] = newshape
        
    return newelement


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
path = 'C:/Users/drikb/Desktop/Land Cover Classifier/Data/'


data = np.load(path + 'lucas_EU_3x3_12GEOMETRIC_MEDIAN.npy',
               allow_pickle = True)

data = np.array([[data[i,0], np.array(np.round(data[i,1]), dtype = 'uint16')] for i in range(len(data))],
                dtype = 'object')


data = np.array(list(map(OrderBands, data)), dtype = 'object')


data = np.array(list(map(Reshape3x3, data)), dtype = 'object')


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

lengths = []
for cat in np.unique(data[:,0]):
    print(len(np.where(data[:,0] == cat)[0]))
    lengths.append(len(np.where(data[:,0] == cat)[0]))
    

    
    
sample_size = min(lengths)
train_index = []
test_index = []
val_index = []
for cat in np.unique(data[:,0]):
    balindex = np.random.choice(np.where(data[:,0] == cat)[0], size = sample_size, replace = False)
    print(len(balindex))
    train_index.extend(balindex[0:int(sample_size*0.60)])
    test_index.extend(balindex[int(sample_size*0.60):int(sample_size*0.80)])
    val_index.extend(balindex[int(sample_size*0.80):-1])
    print(len(train_index), '\n',
          len(test_index), '\n',
          len(val_index), '\n',)
    
train_data, test_data, val_data = data[train_index,:], data[test_index, :], data[val_index, :]

#%%
flipaugment_ver = np.array([np.array([train_data[i,0], np.flip(train_data[i,1], axis = 0)], dtype = 'object') for i in range(len(train_data))])
flipaugment_hor = np.array([np.array([train_data[i,0], np.flip(train_data[i,1], axis = 1)], dtype = 'object') for i in range(len(train_data))])

rotaug_90 = [np.array([train_data[i,0], rotate(train_data[i,1], angle = 90, preserve_range = True).astype('uint16')], dtype = 'object') for i in range(len(train_data))]
rotaug_180 = [np.array([train_data[i,0], rotate(train_data[i,1], angle = 180, preserve_range = True).astype('uint16')], dtype = 'object') for i in range(len(train_data))]
rotaug_270 = [np.array([train_data[i,0], rotate(train_data[i,1], angle = 270, preserve_range = True).astype('uint16')], dtype = 'object') for i in range(len(train_data))]


train_data = np.concatenate([train_data,
                   flipaugment_ver,
                   flipaugment_hor,
                   rotaug_90,
                   rotaug_180,
                   rotaug_270])

#%%

train_data = DataToFlatten(train_data)
test_data = DataToFlatten(test_data)
val_data = DataToFlatten(val_data)

train_x, train_lab = train_data[:,1:], train_data[:,0]
test_x, test_lab = test_data[:,1:], test_data[:,0]
validation_x, validation_lab = val_data[:,1:], val_data[:,0]
    
    
#%%

train_lab -= 1
test_lab -= 1
validation_lab -= 1

#%%
### RF MODEL TRAINING
s = time.time()
parameters = np.empty((0,3))
for regressors in range(1, int(train_x.shape[1]/2 + 5), 5):
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
print('The theoretical accuracy of a random classifier is: ', 1/(len(np.unique(train_lab))))

matrix = confusion_matrix(test_lab, best_pred)
print(matrix.diagonal()/matrix.sum(axis=1))


print(classification_report(test_lab, best_pred))


sns.heatmap(matrix/1000, annot=True)
print(matrix/1000)

#%%

# use this chunk to save the model parameters and avoid tuning every time

bestmodel = RandomForestClassifier(n_estimators = 100, # number of trees
                                  max_samples = 3000, # observations per bootstrapped sample
                                  max_features = 46, # number of regressors
                                  random_state = 42)

bestmodel.fit(train_x, train_lab)

best_pred = bestmodel.predict(test_x)

from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_lab, best_pred))
print('The theoretical accuracy of a random classifier is: ', 1/(len(np.unique(train_lab))))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(test_lab, best_pred)
print(matrix.diagonal()/matrix.sum(axis=1))


from sklearn.metrics import classification_report
print(classification_report(test_lab, best_pred))


import seaborn as sns
sns.heatmap(matrix/1000, annot=True)
print(matrix/1000)


#%%
import time

start_time = time.time()
importances = bestmodel.feature_importances_
std = np.std([tree.feature_importances_ for tree in bestmodel.estimators_], axis=0)
elapsed_time = time.time() - start_time


clust_importances = []
clust_std = []
step = 10
for i in range(0, (3*3*step), step):
    mean = np.mean(importances[i:i+step])
    stand_dev = np.mean(std[i:i+step])
    clust_importances.append(mean)
    clust_std.append(stand_dev)

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

forest_importances = pd.Series(clust_importances, index = ['UL', 'UC', 'UR', 'CL', 'CC', 'CR', 'BL', 'BC', 'BR'])

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=clust_std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


    
