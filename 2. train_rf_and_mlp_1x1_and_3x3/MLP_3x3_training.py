import numpy as np
from skimage.transform import rotate
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from keras import layers
from tensorflow.math import confusion_matrix
from keras import regularizers
import seaborn as sns
from sklearn.metrics import classification_report



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
    rounded = np.round(parameters[:,0], 2)
    max_pos = np.where(rounded == min(rounded))[0]
    best = max_pos[-1]
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


path = 'C:/Users/drikb/Desktop/Tirocinio/EarthEngine/data/'

data = np.load(path + 'lucas_EU_3x3_12M.npy',
               allow_pickle = True)


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


num_classes = len(np.unique(train_lab))


performance = []
for depth in range(6):
    for batch_size in [100, 300, 600, 1000]:
        inputs = layers.Input((train_x.shape[-1]))
        x = layers.Rescaling(1./10000)(inputs)
        added = 0
        while added < depth:
            newfeatures = 128 * (2 ** (depth - added - 1))
            x = layers.Dense(newfeatures, 
                             activation = 'relu',
                             kernel_regularizer = regularizers.L1L2(),
                             bias_regularizer = regularizers.L1L2(),
                             activity_regularizer = regularizers.L1L2())(x)
            added += 1
        x = layers.Dense(64, 
                         activation = 'relu',
                         kernel_regularizer = regularizers.L1L2(),
                         bias_regularizer = regularizers.L1L2(),
                         activity_regularizer = regularizers.L1L2())(x)
        x = layers.Dense(32, 
                         activation = 'relu',
                         kernel_regularizer = regularizers.L1L2(),
                         bias_regularizer = regularizers.L1L2(),
                         activity_regularizer = regularizers.L1L2())(x)
        
    
        x = layers.Dropout(0.5)(x)
    
        outputs = layers.Dense(num_classes, 
                               activation = 'softmax',
                               kernel_regularizer = regularizers.L1L2(),
                               bias_regularizer = regularizers.L1L2(),
                               activity_regularizer = regularizers.L1L2())(x)
    
        model = keras.Model(inputs, outputs)
    
        model.compile(optimizer = 'adam',
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])
        
        model.summary()
        
        callbacks = [keras.callbacks.ModelCheckpoint('NN_compare_3x3.keras',
                                                      save_best_only = True)]

        history = model.fit(train_x, train_lab,
                            epochs = 100,
                            callbacks = callbacks,
                            batch_size = batch_size,
                            validation_data = (validation_x, validation_lab))
        
        test_model = keras.models.load_model('NN_compare_3x3.keras')
        test_loss, test_acc = test_model.evaluate(test_x, test_lab)
        print(f'test accuracy: {test_acc:.3f}')
        
        performance.append([test_acc, test_loss, depth, batch_size])
        
#%%

BestParEvaluator(performance)       

#%%

num_classes = len(np.unique(train_lab))

inputs = layers.Input((train_x.shape[-1]))
x = layers.Rescaling(1./10000)(inputs)
x = layers.Dense(128, 
                 activation = 'relu', 
                 kernel_regularizer = regularizers.L1L2(),
                 bias_regularizer = regularizers.L1L2(),
                 activity_regularizer = regularizers.L1L2())(x)
x = layers.Dense(64, 
                 activation = 'relu',
                 kernel_regularizer = regularizers.L1L2(),
                 bias_regularizer = regularizers.L1L2(),
                 activity_regularizer = regularizers.L1L2())(x)
x = layers.Dense(32, 
                 activation = 'relu',
                 kernel_regularizer = regularizers.L1L2(),
                 bias_regularizer = regularizers.L1L2(),
                 activity_regularizer = regularizers.L1L2())(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, 
                       activation = 'softmax',
                       kernel_regularizer = regularizers.L1L2(),
                       bias_regularizer = regularizers.L1L2(),
                       activity_regularizer = regularizers.L1L2())(x)



model = keras.Model(inputs, outputs)

model.summary()

#%%

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

callbacks = [keras.callbacks.ModelCheckpoint('NN_compare_3x3.keras',
                                             save_best_only = True)]

#%%

history = model.fit(train_x, train_lab,
                    epochs = 150,
                    callbacks = callbacks,
                    batch_size = 300,
                    validation_data = (validation_x, validation_lab))

#%%
test_model = keras.models.load_model('NN_compare_3x3.keras')
test_loss, test_acc = test_model.evaluate(test_x, test_lab)
print(f'test accuracy: {test_acc:.3f}')

#%%
# PLOT AND SAVE THE MODEL IN .png
plot_model(model, 
           to_file='C:/Users/drikb/Downloads/model_plot.png', 
           show_shapes=True, 
           show_layer_names=False)


#%%

predictions = test_model.predict(test_x)
matrix = confusion_matrix(test_lab, predictions.argmax(axis = 1))

sns.heatmap(matrix/1000, annot=True)
print(matrix/1000)

#%%

print(classification_report(test_lab, predictions.argmax(axis = 1)))

