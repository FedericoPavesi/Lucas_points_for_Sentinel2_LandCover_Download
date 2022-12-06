import rasterio
import numpy as np
import pickle
import time
import os
from keras.models import load_model

#%%
def ReadLazio(ref, bands = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B1', 'B9', 'B11', 'B12'],
                return_georef = False):
    image = rasterio.open(ref)
        
    np_image = np.empty((10980, 10980, len(image.descriptions)), dtype = 'uint16')
    
    i = 0
    for band in bands:
        if band == bands[0]:
            print('Processing ', band)
            pos = int(np.where(np.array(image.descriptions) == band)[0])
            firstband = image.read(pos + 1)
            np_image = np.empty((firstband.shape[0], firstband.shape[1], len(bands)), dtype = 'uint16')
            np_image[:,:,i] = firstband
            i += 1
        else:
            print('Processing ', band)
            pos = int(np.where(np.array(image.descriptions) == band)[0])
            np_image[:,:,i] = image.read(pos + 1)
            i += 1
    
    if return_georef == True:
        return np_image, image
    else:
        return np_image
    
def SliceSelect(miny, minx, train_size = 1, dimension = 1098):
    if train_size == 1:
        sb = 0
        sf = 0
    else:
        sb = train_size//2
        sf = train_size//2
    
    bounds = {'miny' : miny - sb,
              'maxy' : miny + dimension + sf,
              'minx' : minx - sb,
              'maxx' : minx + dimension + sf}
    return bounds


def ImageFlattener(image, train_size = 1, return_trace = True):
    if train_size == 1:
        sb = 0
        sf = 0
    else:
        sb = train_size//2
        sf = train_size//2 + 1
    bandsize = train_size**2
    np_raster_flatten = np.empty(((image.shape[0] - (train_size - 1))*(image.shape[1] - (train_size - 1)), bandsize * image.shape[2]), 
                                 dtype = 'uint16')
    
    nonzero = np.where(image[:,:,0] != 0)
        
    image_trace = np.zeros(image.shape[:-1], dtype = 'bool')
        
    i = 0
    step = 1
    for y, x in zip(nonzero[0], nonzero[1]):
        print('step ' , step, ' out of ', len(nonzero[0]))
        if y + sf > image.shape[0] or y - sb < 0 or x + sf > image.shape[1] or x - sb < 0:
            print('Border error at ', y, ' : ', x)
            image_trace[y, x] = False
        elif 0 in image[y-sb:y+sf, x-sb:x+sf,0]:
            print('Zero error at ', y, ' : ', x)
            image_trace[y, x] = False
        elif image[y, x, 0] == 0:
            print('Zero error at ', y, ' : ', x)
            image_trace[y, x] = False
        else:
            if train_size == 1:
                #a = np.concatenate((image[y, x, : ].flatten(), [y], [x]))
                a = image[y,x,:].flatten()
            else:
                #a = np.concatenate((image[y-border:y+border, x-border:x+border, : ].flatten(), [y], [x]))
                a = image[y-sb:y+sf, x-sb:x+sf, : ].flatten()
            np_raster_flatten[i,:] = a
            image_trace[y, x] = True
            i += 1
        step += 1
    if return_trace == True:
        return np_raster_flatten, image_trace
    else:
        return np_raster_flatten
    


    
    
def FastImageFlattener(image, return_trace = True):
    
    nonzero = np.where(image[:,:,0] != 0)
    
    image_trace = np.zeros((image.shape[0], image.shape[1]), dtype = 'bool')
    image_trace[nonzero] = True
    
    np_raster_flatten = image[nonzero[0], nonzero[1], :].reshape((len(nonzero[0]), image.shape[-1]))
        
    
    if return_trace == True:
        return np_raster_flatten, image_trace
    else:
        return np_raster_flatten
    
    
def KernelImageFlattener(image, kernel_size = 3, return_trace = True, padding_mode = 'edge'):
    pad_size = kernel_size // 2
    pad = np.pad(image, [(pad_size,pad_size), (pad_size,pad_size), (0,0)], mode = padding_mode)
    
    nonzero = np.where(image[:,:,0] != 0)
    
    image_trace = np.zeros((image.shape[0], image.shape[1]), dtype = 'bool')
    image_trace[nonzero] = True
        
    step_fw = kernel_size // 2 + 1 + pad_size #we do this because nonzero positions doesn't take into
    step_bw = -(kernel_size // 2) + pad_size     #account the padding
    
    kernels = [pad[i + step_bw: i + step_fw, j + step_bw: j + step_fw] for i, j in zip(nonzero[0], nonzero[1])]

    np_raster_flatten = np.array([kernels[i].flatten() for i in range(len(kernels))])

    if return_trace == True:
        return np_raster_flatten, image_trace
    else:
        return np_raster_flatten


def PredictionsToImage(predictions, trace, train_size = 1):
    finmap = np.zeros((trace.shape[0], trace.shape[1]), dtype = 'uint8')
    nonzero = np.where(trace == True)
    i = 0
    for y,x in zip(nonzero[0], nonzero[1]):
        finmap[y, x] = predictions[i]
        i += 1
    return finmap


def SaveToGeoTiff(save_directory, map_data, reference):
    transform_to = reference.transform
    
    new_dataset = rasterio.open(save_directory, 'w', driver='GTiff',
                                height = map_data.shape[0], width = map_data.shape[1],
                                count=1, dtype=str(map_data.dtype),
                                crs= reference.crs,
                                transform=transform_to)
    new_dataset.write(map_data, 1)
    new_dataset.close()



#%%
###############################################################################
#               RANDOM FOREST                                                 #
###############################################################################

modelpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Models/RF/'
rasterpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Rasters/'

pred_times = []

for raster_name in os.listdir(rasterpath):
    t0 = time.time()

    image, ref = ReadLazio(ref = rasterpath + raster_name, return_georef = True)
    np_raster_flatten, image_trace = KernelImageFlattener(image)
    print('Image correctly flattened!')
    
    try:
        for model in os.listdir(modelpath)[:2]:
            if '3x3' in model:
                bestmodel = pickle.load(open(modelpath + model, 'rb'))
                print('Model loaded!')
                savepath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Predictions/RF3X3/trial_' + model[-1] + '/'
                
                if 'Lazio_pred_' + raster_name[-25:] in os.listdir(savepath):
                    continue
                
                print('Start prediction...')
                predictions_map = bestmodel.predict(np_raster_flatten)
                print('END!')
                predictions_map += 1
                
                finmap = PredictionsToImage(predictions_map, trace = image_trace, train_size = 1)
                
                SaveToGeoTiff(savepath + 'Lazio_pred_' + raster_name[-25:], finmap, ref)
    except:
        print('Error: raster is empty, proceding to next one...')
        
    t1 = time.time()
    print('Process took', (t1 - t0)/60, ' minutes')
    pred_times.append((t1-t0)/60)



#%%

###############################################################################
#                MLP                                                          #
###############################################################################


modelpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Models/MLP/'
rasterpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Rasters/'

pred_times = []

for raster_name in os.listdir(rasterpath):
    t0 = time.time()

    image, ref = ReadLazio(ref = rasterpath + raster_name, return_georef = True)
    
    if raster_name == 'Lazio_2018-0000000000-0000000000-002.tif':
        print('Elaborating large raster')
        kernel_size = 3
        padding_mode = 'edge'
        pad_size = kernel_size // 2
        pad = np.pad(image, [(pad_size,pad_size), (pad_size,pad_size), (0,0)], mode = padding_mode)
        
        nonzero = np.where(image[:,:,0] != 0)
        
        image_trace = np.zeros((image.shape[0], image.shape[1]), dtype = 'bool')
        image_trace[nonzero] = True
            
        step_fw = kernel_size // 2 + 1 + pad_size #we do this because nonzero positions doesn't take into
        step_bw = -(kernel_size // 2) + pad_size     #account the padding
        
        kernels = [pad[i + step_bw: i + step_fw, j + step_bw: j + step_fw] for i, j in zip(nonzero[0], nonzero[1])]
        
        data_step_split = int(len(kernels)/4) #splitting process_____
        
        try:
            for model in os.listdir(modelpath):
                if '3x3' in model:
                    bestmodel = load_model(modelpath + model)
                    print('Model loaded!')
                    
                    savepath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Predictions/MLP3x3/trial_' + model[-1] + '/'
                    
                    if 'Lazio_pred_' + raster_name[-25:] in os.listdir(savepath):
                        continue
                    
                    finmap = []
                    
                    for split_step in range(4): #splitting process_____
                        print('processing from', split_step*data_step_split, 'to', (split_step+1)*data_step_split)
                        if split_step == 3: #last iteration_____
                            np_raster_flatten = np.array([kernels[i].flatten() for i in range(split_step*data_step_split, len(kernels))], dtype = 'uint16')
                        else:
                            np_raster_flatten = np.array([kernels[i].flatten() for i in range(split_step*data_step_split, (split_step + 1)*data_step_split)], dtype = 'uint16')
                        
                        print('Start prediction...')
                        predictions_map = bestmodel.predict(np_raster_flatten).argmax(axis = 1)
                        print('END!')
                        
                        predictions_map += 1
                        
                        finmap.append(predictions_map)
                    
                    print('merging predictions...')
                    finmap = np.concatenate(finmap).astype('uint8')
                    
                    final_map = PredictionsToImage(finmap, trace = image_trace, train_size = 1)
                     
                    SaveToGeoTiff(savepath + 'Lazio_pred_' + raster_name[-25:], final_map, ref)
                        
                    t1 = time.time()
                    print('Process took', (t1 - t0)/60, ' minutes')
                    pred_times.append((t1-t0)/60)
                    
                    del finmap, final_map, np_raster_flatten
                    
            del kernels
        
        except KeyboardInterrupt:
            break
        except:
            print('Error: raster is empty, proceding to next one...')
    else:
        np_raster_flatten, image_trace = KernelImageFlattener(image)
        try:
            for model in os.listdir(modelpath):
                if '3x3' in model:
                    bestmodel = load_model(modelpath + model)
                    print('Model loaded!')
        
                    savepath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Predictions/MLP3x3/trial_' + model[-1] + '/'
                    
                    if 'Lazio_pred_' + raster_name[-25:] in os.listdir(savepath):
                        continue
                    
                    print('Start prediction...')
                    predictions_map = bestmodel.predict(np_raster_flatten).argmax(axis = 1)
                    print('END!')
        
                    predictions_map += 1
                    
                    finmap = PredictionsToImage(predictions_map, trace = image_trace, train_size = 1)
                    
                    SaveToGeoTiff(savepath + 'Lazio_pred_' + raster_name[-25:], finmap, ref)
        except KeyboardInterrupt:
            break
        except:
            print('Error: raster is empty, proceding to next one...')
            
        t1 = time.time()
        print('Process took', (t1 - t0)/60, ' minutes')
        pred_times.append((t1-t0)/60)
        
        
#%%


############ OPTIMIZED RF

modelpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Models/RF/'
rasterpath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Rasters/'

pred_times = []

max_models = 2


n_models = 0

for raster_name in os.listdir(rasterpath):
    t0 = time.time()

    image, ref = ReadLazio(ref = rasterpath + raster_name, return_georef = True)
    
    print('Elaborating large raster')
    kernel_size = 3
    padding_mode = 'edge'
    pad_size = kernel_size // 2
    pad = np.pad(image, [(pad_size,pad_size), (pad_size,pad_size), (0,0)], mode = padding_mode)
    
    nonzero = np.where(image[:,:,0] != 0)
    
    image_trace = np.zeros((image.shape[0], image.shape[1]), dtype = 'bool')
    image_trace[nonzero] = True
        
    step_fw = kernel_size // 2 + 1 + pad_size #we do this because nonzero positions doesn't take into
    step_bw = -(kernel_size // 2) + pad_size     #account the padding
    
    kernels = [pad[i + step_bw: i + step_fw, j + step_bw: j + step_fw] for i, j in zip(nonzero[0], nonzero[1])]
    
    data_step_split = int(len(kernels)/40) #splitting process_____
    
    try:
        for model in os.listdir(modelpath):
            if '3x3' in model:
                # if n_models == max_models: #___________________
                #     break
                bestmodel = pickle.load(open(modelpath + model, 'rb'))
                # n_models += 1  #________________

                print('Model loaded!')
                
                savepath = 'C:/Users/drikb/Desktop/Land Cover Classifier/Predictions/RF3x3/trial_' + model[-1] + '/'
                
                if 'Lazio_pred_' + raster_name[-25:] in os.listdir(savepath):
                    continue
                
                finmap = []
                
                for split_step in range(40): #splitting process_____
                    print('processing from', split_step*data_step_split, 'to', (split_step+1)*data_step_split)
                    if split_step == 39: #last iteration_____
                        np_raster_flatten = np.array([kernels[i].flatten() for i in range(split_step*data_step_split, len(kernels))], dtype = 'uint16')
                    else:
                        np_raster_flatten = np.array([kernels[i].flatten() for i in range(split_step*data_step_split, (split_step + 1)*data_step_split)], dtype = 'uint16')
                    
                    print('Start prediction...')
                    predictions_map = bestmodel.predict(np_raster_flatten)
                    print('END!')
                    
                    predictions_map += 1
                    
                    finmap.append(predictions_map)
                
                print('merging predictions...')
                finmap = np.concatenate(finmap).astype('uint8')
                
                final_map = PredictionsToImage(finmap, trace = image_trace, train_size = 1)
                 
                SaveToGeoTiff(savepath + 'Lazio_pred_' + raster_name[-25:], final_map, ref)
                    
                t1 = time.time()
                print('Process took', (t1 - t0)/60, ' minutes')
                pred_times.append((t1-t0)/60)
                
                del finmap, final_map, np_raster_flatten
                
        del kernels
    
    except KeyboardInterrupt:
        break
    except:
        print('Error: raster is empty, proceding to next one...')
    

