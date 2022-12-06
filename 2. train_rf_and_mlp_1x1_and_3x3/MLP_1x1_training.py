import numpy as np
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

def OrderBands(element, from_order = ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9'], to_order = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B1', 'B9', 'B11', 'B12']):
    from_order = np.array(from_order)
    to_order = np.array(to_order)
    image = element[1]
    newimage = np.empty((image.shape[0], image.shape[1], image.shape[2]), dtype = 'uint16')
    for band_pos in range(len(to_order)):
        band = np.where(from_order == to_order[band_pos])[0]
        newimage[band_pos,:,:] = image[band,:,:]
        
    return np.array([element[0], newimage], dtype = 'object')
        


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


# we select central pixel of 3x3 images
data = np.array([[data[i,0], data[i,1][:,1,1].reshape((1,1,12))] for i in range(len(data))], dtype = 'object')

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



train_x, test_x, validation_x = df[train_index], df[test_index], df[val_index]
train_lab, test_lab, validation_lab = labels[train_index], labels[test_index], labels[val_index]



     
    
#%%

# MLP requires labels starting from 0

train_lab -= 1
test_lab -= 1
validation_lab -= 1
    

#%%


num_classes = len(np.unique(labels))


performance = []
for depth in range(6):
    for batch_size in [300, 600, 1000]:
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
        
        callbacks = [keras.callbacks.ModelCheckpoint('NN_compare_1x1.keras',
                                                      save_best_only = True)]

        history = model.fit(train_x, train_lab,
                            epochs = 300,
                            callbacks = callbacks,
                            batch_size = batch_size,
                            validation_data = (validation_x, validation_lab))
        
        test_model = keras.models.load_model('NN_compare_1x1.keras')
        test_loss, test_acc = test_model.evaluate(test_x, test_lab)
        print(f'test accuracy: {test_acc:.3f}')
        
        performance.append([test_loss, test_acc, depth, batch_size])
        
#%%

BestParEvaluator(np.array(performance))    

#%%

num_classes = len(np.unique(labels))

inputs = layers.Input((df.shape[-1]))
x = layers.Rescaling(1./10000)(inputs)
x = layers.Dense(256, 
                  activation = 'relu', 
                  kernel_regularizer = regularizers.L1L2(),
                  bias_regularizer = regularizers.L1L2(),
                  activity_regularizer = regularizers.L1L2())(x)
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

callbacks = [keras.callbacks.ModelCheckpoint('NN_compare_1x1.keras',
                                             save_best_only = True)]

#%%

history = model.fit(train_x, train_lab,
                    epochs = 600,
                    callbacks = callbacks,
                    batch_size = 300,
                    validation_data = (validation_x, validation_lab))

#%%
test_model = keras.models.load_model('NN_compare_1x1.keras')
test_loss, test_acc = test_model.evaluate(test_x, test_lab)
print(f'test accuracy: {test_acc:.3f}')

#%%

# PLOT AND SAVE THE MODEL IN .png
plot_model(model, 
           to_file='C:/Users/drikb/Desktop/Land Cover Classifier/Images/MLP1x1_model_plot.png', 
           show_shapes=True, 
           show_layer_names=False)



#%%

predictions = test_model.predict(test_x)
matrix = confusion_matrix(test_lab, predictions.argmax(axis = 1))

sns.heatmap(matrix, annot=True)
print(matrix)

#%%

print(classification_report(test_lab, predictions.argmax(axis = 1)))
