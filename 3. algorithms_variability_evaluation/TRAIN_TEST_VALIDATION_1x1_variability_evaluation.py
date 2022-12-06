import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from tensorflow.math import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import regularizers
import aspose.words as aw

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

def OrderBands(element, from_order = ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9'], to_order = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B1', 'B9', 'B11', 'B12']):
    from_order = np.array(from_order)
    to_order = np.array(to_order)
    image = element[1]
    newimage = np.empty((image.shape[0], image.shape[1], image.shape[2]), dtype = 'uint16')
    for band_pos in range(len(to_order)):
        band = np.where(from_order == to_order[band_pos])[0]
        newimage[band_pos,:,:] = image[band,:,:]
        
    return np.array([element[0], newimage], dtype = 'object')

def SaveWordTable(path, numpy_table):
    nrow = numpy_table.shape[0]
    ncol = numpy_table.shape[1]
    
    doc = aw.Document()    
    builder = aw.DocumentBuilder(doc)
    
    table = builder.start_table()
    
    builder.insert_cell()
    
    table.left_indent = 20.0
    
    builder.row_format.height = 40.0
    
    builder.paragraph_format.alignment = aw.ParagraphAlignment.CENTER
    
    builder.font.size = 12
    
    builder.font.name = 'Times New Roman'
    
    builder.cell_format.width = 100.0
    
    builder.write(str(np.round(numpy_table[0,0], 4)))
    
    for row in range(nrow):
        for col in range(ncol):
            if row == 0 and col == 0:
                continue
            elif col+1 == ncol:
                builder.insert_cell()
                builder.write(str(np.round(numpy_table[row, col], 4)))
                builder.end_row()
            else:
                builder.insert_cell()
                builder.write(str(np.round(numpy_table[row, col], 4)))
                
    builder.end_table()
    
    doc.save(path)

#%%


## LUCAS CATEGORIES DICTIONARY
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

for cat in range(len(code)):
    index = np.where(data[:,0] == code[cat])[0]
    data[index, 0] = number[cat]

#%%

###############################################################################
#                       MLP                                                   #
###############################################################################

modelpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Models/MLP/'


MLP_comparisons = []

for trial in range(10):
    len(np.where(data[:,0] == 1)[0])
    
    np.random.shuffle(data)
    
    
    reshapeddata = DataToFlatten(data)
    
    for cat in np.unique(reshapeddata[:,0]):
        print(len(np.where(reshapeddata[:,0] == cat)[0]))
    
    np.random.shuffle(reshapeddata)
    
    lengths = []
    for cat in np.unique(reshapeddata[:,0]):
        print(len(np.where(reshapeddata[:,0] == cat)[0]))
        lengths.append(len(np.where(reshapeddata[:,0] == cat)[0]))
        
    df = reshapeddata[:,1:]
    labels = reshapeddata[:,0]
    
    
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
    
    
    
    for cat in np.unique(train_lab):
        print(len(np.where(train_lab == cat)[0]))
    
    
    train_lab -= 1
    test_lab -= 1
    validation_lab -= 1
        
    
    
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
    
    
    model.compile(optimizer = 'adam',
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    callbacks = [keras.callbacks.ModelCheckpoint('RF_NN_compare_1x1.keras',
                                                 save_best_only = True)]
    
    
    history = model.fit(train_x, train_lab,
                        epochs = 600,
                        callbacks = callbacks,
                        batch_size = 300,
                        validation_data = (validation_x, validation_lab))
    
    
    test_model = keras.models.load_model('RF_NN_compare_1x1.keras')
    test_loss, test_acc = test_model.evaluate(test_x, test_lab)
    print(f'test accuracy: {test_acc:.3f}')
    
    test_model.save(modelpath + 'NN_1x1_trial_' + str(trial))
    
    predictions = test_model.predict(test_x)
    matrix = confusion_matrix(test_lab, predictions.argmax(axis = 1))
    matrix = matrix.numpy()
    class_rep = classification_report(test_lab, predictions.argmax(axis = 1))
    MLP_comparisons.append([matrix, class_rep])


#%%
savepath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Models_Performances/MLP/'

np.save(savepath + 'MLP_1x1_performances.npy',
        np.array(MLP_comparisons, dtype = 'object'))


# MLP_comparisons = np.load(savepath + 'MLP_1x1_performances.npy',
#                             allow_pickle = True)

        
#%%
MLP_comparisons = np.array(MLP_comparisons, dtype = 'object')
MLP_comparisons[:,1]
conf_perc = MLP_comparisons[:,0]/1000
mean_mlp_conf_mat = np.mean(conf_perc, axis = 0)
std_mlp_conf_mat = np.std(conf_perc, axis = 0)

avg_accuracy = np.mean(mean_mlp_conf_mat.diagonal())


SaveWordTable(savepath + 'MLP1x1_mean_conf_mat.docx', mean_mlp_conf_mat)         
SaveWordTable(savepath + 'MLP1x1_std_conf_mat.docx', std_mlp_conf_mat)         
          

#%%

###############################################################################
#              RANDOM FOREST                                                  #
###############################################################################
modelpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Models/RF/'

RF_comparisons = []
for trial in range(10):
    len(np.where(data[:,0] == 1)[0])
    
    np.random.shuffle(data)
    
    
    reshapeddata = DataToFlatten(data)
    
    for cat in np.unique(reshapeddata[:,0]):
        print(len(np.where(reshapeddata[:,0] == cat)[0]))
    
    np.random.shuffle(reshapeddata)
    
    lengths = []
    for cat in np.unique(reshapeddata[:,0]):
        print(len(np.where(reshapeddata[:,0] == cat)[0]))
        lengths.append(len(np.where(reshapeddata[:,0] == cat)[0]))
        
    df = reshapeddata[:,1:]
    labels = reshapeddata[:,0]
        
    
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
    
    
    
    for cat in np.unique(train_lab):
        print(len(np.where(train_lab == cat)[0]))
        
    bestmodel = RandomForestClassifier(n_estimators = 100, # number of trees
                                      max_samples = 3000, # observations per bootstrapped sample
                                      max_features = 10, # number of regressors
                                      random_state = 42)
    
    train_lab -= 1
    test_lab -= 1
    validation_lab -= 1
    
    bestmodel.fit(train_x, train_lab)
    
    best_pred = bestmodel.predict(test_x)
    
    
    print("Accuracy = ", metrics.accuracy_score(test_lab, best_pred))
    print('The theoretical accuracy of a random classifier is: ', 1/(len(np.unique(labels))))
    
    pickle.dump(bestmodel, open(modelpath + 'RF_1x1_trial_' + str(trial), 'wb'))
    
    matrix = confusion_matrix(test_lab, best_pred)
    matrix = matrix.numpy()
    class_rep = classification_report(test_lab, best_pred)
    RF_comparisons.append([matrix, class_rep])
    
#%%

savepath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Models_Performances/RF/'

np.save(savepath + 'RF_1x1_performances.npy',
        np.array(RF_comparisons, dtype = 'object'))


# RF_comparisons = np.load(savepath + 'RF_1x1_performances.npy',
#                             allow_pickle = True)

#%%
RF_comparisons = np.array(RF_comparisons, dtype = 'object')
conf_perc = RF_comparisons[:,0]/1000
mean_rf_conf_mat = np.mean(conf_perc, axis = 0)
std_rf_conf_mat = np.std(conf_perc, axis = 0)

avg_accuracy = np.mean(mean_rf_conf_mat.diagonal())


SaveWordTable(savepath + 'RF1x1_mean_conf_mat.docx', mean_rf_conf_mat)         
SaveWordTable(savepath + 'RF1x1_std_conf_mat.docx', std_rf_conf_mat)         
 