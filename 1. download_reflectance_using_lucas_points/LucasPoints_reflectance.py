
import pandas as pd
import numpy as np
import ee
import datetime
from dateutil.relativedelta import relativedelta
import time
import functools

#%%
ee.Initialize()

#%%

###############   FUNCTION DEFINITION   #######################################

def DateToQuery(date, months = 1, days = 0, dateformat = '%d/%m/%y', new_dateformat = '%Y-%m-%d'):
    dateformatted = datetime.datetime.strptime(date, dateformat)
    start_date = dateformatted - relativedelta(months = months, days = days)
    end_date = dateformatted + relativedelta(months = months, days = days)
    start_date = datetime.datetime.strftime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strftime(end_date, '%Y-%m-%d')
    return start_date, end_date


def ShiftDate(datecol, shift = 6, dateformat = '%d/%m/%y'):
    dateformatted = datetime.datetime.strptime(datecol, dateformat)
    if shift > 0:
        newdate = dateformatted + relativedelta(months = shift, days = 0)
    elif shift < 0:
        shift = abs(shift)
        newdate = dateformatted - relativedelta(months = shift, days = 0)
    else:
        print('Error, no shift selected, returning original date')
        return datecol
    newdate = datetime.datetime.strftime(newdate, dateformat)
    return newdate


def AugmentLucasFrame(pdframe, categories, length, digits = 1, shift = 6, 
                      dateformat = '%d/%m/%y', max_interval = 8):
    augment_list = []
    for cat in categories:
        index = np.where(pdframe['LC1'].str[digits-1] == cat)[0]
        to_augment = length - len(index)
        print('We need to add ', to_augment, 'elements for ', cat)
        if to_augment <= 0:
            print('Category already enough represented, skipping...')
            continue
        
        if to_augment/len(index) < 0:
            chosen = np.random.choice(index, size = to_augment, replace = False)
            print('Chosen done...')
            for i in chosen:
                newdate = ShiftDate(pdframe.loc[i,pdframe.columns[6]], shift = shift,
                                    dateformat = dateformat)
                val = {pdframe.columns[1] : pdframe.loc[i,pdframe.columns[1]],
                       pdframe.columns[2] : pdframe.loc[i,pdframe.columns[2]],
                       pdframe.columns[3] : pdframe.loc[i,pdframe.columns[3]],
                       pdframe.columns[4] : pdframe.loc[i,pdframe.columns[4]],
                       pdframe.columns[5] : pdframe.loc[i,pdframe.columns[5]],
                       pdframe.columns[6] : newdate}
                augment_list.append(val)
        else:
            augment_number = int(to_augment/len(index)) + 1
            if augment_number > max_interval:
                print('For category ', cat, ' we are not able to reach the objective \n', 'The maximum reachable number is ', max_interval*len(index))
                to_augment = max_interval*len(index)
                augment_number = max_interval
            for i in index:
                for shift_num in range(1, augment_number+1):
                    if shift_num%2 == 1:
                        newdate = ShiftDate(pdframe.loc[i,pdframe.columns[6]], shift = shift*(shift_num//2 + 1),
                                            dateformat = dateformat)
                        val = {pdframe.columns[1] : pdframe.loc[i,pdframe.columns[1]],
                               pdframe.columns[2] : pdframe.loc[i,pdframe.columns[2]],
                               pdframe.columns[3] : pdframe.loc[i,pdframe.columns[3]],
                               pdframe.columns[4] : pdframe.loc[i,pdframe.columns[4]],
                               pdframe.columns[5] : pdframe.loc[i,pdframe.columns[5]],
                               pdframe.columns[6] : newdate}
                        augment_list.append(val)
                    else:
                        newdate = ShiftDate(pdframe.loc[i,pdframe.columns[6]], shift = -shift*(shift_num//2),
                                            dateformat = dateformat)
                        val = {pdframe.columns[1] : pdframe.loc[i,pdframe.columns[1]],
                               pdframe.columns[2] : pdframe.loc[i,pdframe.columns[2]],
                               pdframe.columns[3] : pdframe.loc[i,pdframe.columns[3]],
                               pdframe.columns[4] : pdframe.loc[i,pdframe.columns[4]],
                               pdframe.columns[5] : pdframe.loc[i,pdframe.columns[5]],
                               pdframe.columns[6] : newdate}
                        augment_list.append(val)
    
    augment_frame = pd.DataFrame.from_records(augment_list,
                                              columns = pdframe.columns[1:])
    return augment_frame



def FlagCloudFree(image, point):
    
    
    proj = image.select('B4').projection()
    
    neigh = image.neighborhoodToArray(ee.Kernel.square(step), 0)
    
    ext = neigh.reduceRegion(reducer = ee.Reducer.first(),
                             geometry = point,
                             crs = proj,
                             scale = proj.nominalScale())
        
    free_opaque = ee.Algorithms.If(ext.getArray('QA60').toList().flatten().contains(1024),0,1)
    free_cirrus = ee.Algorithms.If(ext.getArray('QA60').toList().flatten().contains(2048),0,1)
    
    free = ee.Number(free_opaque).multiply(ee.Number(free_cirrus))
    
    return image.set({'pixel_cloud_free' : free})

def NullRemover2(image):
    proj = image.select('B4').projection()
    
    neigh = image.neighborhoodToArray(ee.Kernel.square(step), 0)
    
    ext = neigh.reduceRegion(reducer = ee.Reducer.first(),
                              geometry = aoi,
                              crs = proj,
                              scale = proj.nominalScale())
    
    emptdict = {'AOT': None,
     'B1': None,
     'B11': None,
     'B12': None,
     'B2': None,
     'B3': None,
     'B4': None,
     'B5': None,
     'B6': None,
     'B7': None,
     'B8': None,
     'B8A': None,
     'B9': None,
     'MSK_CLDPRB': None,
     'MSK_SNWPRB': None,
     'QA10': None,
     'QA20': None,
     'QA60': None,
     'SCL': None,
     'TCI_B': None,
     'TCI_G': None,
     'TCI_R': None,
     'WVP': None}
    
    isempty = ee.Algorithms.IsEqual(ext, ee.Dictionary(emptdict))
    
    return image.set({'is_empty' : isempty})


def NullRemover(image):
    proj = image.select('B4').projection()
    
    neigh = image.neighborhoodToArray(ee.Kernel.square(step), 0)
    
    ext = neigh.reduceRegion(reducer = ee.Reducer.first(),
                              geometry = aoi,
                              crs = proj,
                              scale = proj.nominalScale())
    
    B4_dict = ee.Dictionary({'B4' : ee.List(ext.get('B4'))})
    
    check_dict = ee.Dictionary({'B4' : None})
    
    isempty = ee.Algorithms.IsEqual(B4_dict, check_dict)
        
    return image.set({'is_empty' : isempty})

#%%

################## LUCAS 2018 COUNTRIES MERGE  ################################

#VARIABLES
lucas_path = 'C:/Users/drikb/Desktop/Tirocinio/GRAPH NEURAL NETWORKS/Random forest project/Dati/LUCAS/lucas_points_2018/'
countries = ['AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 
             'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 'NL', 'PL', 
             'PT', 'RO', 'SE', 'SI', 'SK', 'UK']
variables_to_keep = ['POINT_ID', 'GPS_PROJ', 'TH_LAT', 'TH_LONG', 'LC1', 'SURVEY_DATE']


datalist = []
for country in countries:
    lucas_country = pd.read_csv(lucas_path + country + '_2018_20200213.CSV')
    lucas_country = lucas_country[variables_to_keep]
    lucas_country = lucas_country[lucas_country['GPS_PROJ'] == 1]
    lucas_country = lucas_country[lucas_country['LC1'] != '8']
    lucas_country = lucas_country.reset_index()
    datalist.append(lucas_country)
    
lucas = pd.concat(datalist)

lucas = lucas.drop('index', axis = 1)
lucas = lucas.reset_index()


#%%

############# UNDERREPRESENTED CLASSES AUGMENTATION ###########################

# VARIABLES
aug_len = 5000
max_interval = 12 # max shifted months


lucas_aug = AugmentLucasFrame(lucas, categories = ['A', 'E', 'D', 'F', 'G', 'H'], length = aug_len, max_interval = max_interval)
lucas_aug = lucas_aug.reset_index()

lucas = pd.concat([lucas, lucas_aug], ignore_index = True)
lucas = lucas.reset_index()
lucas = lucas.drop(['level_0', 'index'], axis = 1)


for cat in np.unique(lucas['LC1'].str[0]):
    print(cat)

#%%

############### RANDOM SAMPLE #################################################

# VARIABLES
size = 5000 # notice: it is advisable to have size <= aug_len


select_index = []

for cat in np.unique(lucas['LC1'].str[0]):
    print(cat)
    where = np.where(lucas['LC1'].str[0] == cat)[0]
    selected = np.random.choice(where, size = size, replace = False)
    select_index.extend(selected)
    

lucas = lucas.loc[select_index]
lucas = lucas.reset_index(drop = True)


#%%

############ DEFINE VARIABLES FOR DOWNLOAD STEP ###############################

# shape of downloaded image. Notice it must be always a square (ex. 3x3) as we
# are using a kernel to define the image size. In any case, the code considers 
# the first element of the shape.
shape = (3, 3)

# maximum cloud cover percentage in considered images
cloud_filter = 60

# bands we are interested in downloading
bands = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B1', 'B9', 'B11', 'B12']

# time frame on which to consider the median reflectance
months = 2 
days = 0

# buffer_size selects the number of features to collect before using getInfo() 
# method. This is done to speed up the process (in 3x3 images we go from 12 hours
# to 1h30m), if you get a 'limit exceeded' error from earth engine you should 
# lower the buffer size until everything is fine. As much the buffer size is high, 
# as much the process is fastened.
buffer_size = 100

#%%

ext_order = ['B1', 'B11', 'B12', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9']
#%%

# VARIABLES
lat_var = 'TH_LAT'
long_var = 'TH_LONG'
date_var = 'SURVEY_DATE'
lc_var = 'LC1'



s = time.time()
output_collection = []
errors_index = []

ee_points_collection = []
point = 30000
while point < len(lucas):   
      
    step = shape[0]//2
        
    lat = lucas.loc[point, lat_var]
    long = lucas.loc[point, long_var]
    date = lucas.loc[point, date_var]
    lc1 = lucas.loc[point, lc_var]

    aoi = ee.Geometry.Point([long, lat], 'EPSG:4326')
    start_date, end_date = DateToQuery(date, months = months, days = days)

    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date))
    
    s2_sr_col = s2_sr_col.map(NullRemover)

    s2_sr_col = s2_sr_col.filter(ee.Filter.eq('is_empty', False))    
    
    individual_flag = functools.partial(FlagCloudFree, 
                                        point = aoi)

    s2_sr_col = s2_sr_col.map(individual_flag)

    s2_sr_col = s2_sr_col.filter(ee.Filter.eq('pixel_cloud_free',1))
        

    projection = s2_sr_col.first().select('B4').projection() 
    
    
    img = s2_sr_col.select(bands).reduce(ee.Reducer.geometricMedian(len(bands)).setOutputs(bands))
    
    neighborhood = img.neighborhoodToArray(ee.Kernel.square(step), 0)
        
    extracted = neighborhood.reduceRegion(reducer = ee.Reducer.first(),
                                          geometry = aoi,
                                          crs = projection,
                                          scale = projection.nominalScale())
    
    listobj = ee.List([lc1, extracted.values()])

   
    ee_points_collection.append(listobj)   
    
    if (point+1)%buffer_size == 0:
        st = time.time()
        print('processing point ', str(point + 1), ' out of ', str(len(lucas)))
        ee_points_collection = ee.List(ee_points_collection)
        output_collection.extend(ee_points_collection.getInfo())
        en = time.time()
        print('Iteration took', en-st, 'seconds')
        ee_points_collection = []
    elif point+1 == len(lucas):
        ee_points_collection = ee.List(ee_points_collection)
        output_collection.extend(ee_points_collection.getInfo())

    point += 1
    
e = time.time()

print('Download took ', e - s,' seconds.')

#%%

# function currentHS_function(image){
#   var proxy = -999
#   image = image.unmask(proxy)
#   var currentHS = image.expression(
#  "(b('current') == proxy) ? 0.0" +
#  ": (b('current') < 0.13) ? 1.0" +
#  ": (b('current') > 0.15) ? 0.0" +
#  ": -50.0*current + 15.0/2.0 ",
#  {"current" : image, "proxy": proxy});
#    return currentHS;
#  }
#%%

point = 30000
while point < len(lucas): 
    
    print(point)
    
    point = 30010
      
    step = shape[0]//2
        
    lat = lucas.loc[point, lat_var]
    long = lucas.loc[point, long_var]
    date = lucas.loc[point, date_var]
    lc1 = lucas.loc[point, lc_var]

    aoi = ee.Geometry.Point([long, lat], 'EPSG:4326')
    start_date, end_date = DateToQuery(date, months = months, days = days)

    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
                 .filterBounds(aoi)
                 .filterDate(start_date, end_date))
    
    s2_sr_col.size().getInfo()
    
    # s2list = s2_sr_col.toList(s2_sr_col.size())
    
    # size = s2_sr_col.size().getInfo()
    
    # i = 0
    # while i < size:
                
    #     image = ee.Image(s2list.get(i))
        
    #     proj = image.select('B4').projection()
        
    #     neigh = image.neighborhoodToArray(ee.Kernel.square(step), 0)
        
    #     ext = neigh.reduceRegion(reducer = ee.Reducer.first(),
    #                               geometry = aoi,
    #                               crs = proj,
    #                               scale = proj.nominalScale())
        
    #     ext.getInfo()
        
    #     B4_dict = ee.Dictionary({'B4' : ee.List(ext.get('B4'))})
        
    #     check_dict = ee.Dictionary({'B4' : None})
        
    #     isempty = ee.Algorithms.IsEqual(B4_dict, check_dict)
        
    #     print('\n', i, 'is fine')
    #     if isempty.getInfo() == False:
    #         print(ext.getInfo())
        
    #     i += 1
        
    
    s2_sr_col = s2_sr_col.map(NullRemover2)
    
    s2_sr_col.size().getInfo()

    s2_sr_col = s2_sr_col.filter(ee.Filter.eq('is_empty', False))  
    
    s2_sr_col.size().getInfo()
    
    individual_flag = functools.partial(FlagCloudFree, 
                                        point = aoi)

    s2_sr_col = s2_sr_col.map(individual_flag)

    s2_sr_col = s2_sr_col.filter(ee.Filter.eq('pixel_cloud_free',1))
    
    #s2_sr_col.size().getInfo()
        

    projection = s2_sr_col.first().select('B4').projection() 
    
    
    img = s2_sr_col.select(bands).reduce(ee.Reducer.geometricMedian(len(bands)).setOutputs(bands))
    
    neighborhood = img.neighborhoodToArray(ee.Kernel.square(step), 0)
        
    extracted = neighborhood.reduceRegion(reducer = ee.Reducer.first(),
                                          geometry = aoi,
                                          crs = projection,
                                          scale = projection.nominalScale())
    
    extracted.getInfo()
    
    point += 1
    
    
#%%

# define a path where to save 
save_path = 'C:/Users/drikb/Desktop/Land Cover Classifier/Data/'

# check to ensure there are no missing data
cleaned = [i for indx, i in enumerate(output_collection) if type(output_collection[indx][1]) != type(None)]

cleaned == output_collection

# transform into numpy array
cleaned = np.array(cleaned, dtype = 'object')

# save numpy array
np.save(save_path + 'lucas_EU_' + str(shape[0]) + 'x' + str(shape[1]) + 
        '_' + str(len(bands)) + 'GEOMETRIC_MEDIAN.npy', cleaned)    
