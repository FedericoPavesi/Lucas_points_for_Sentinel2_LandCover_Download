
###############################################################################
#   NOTICE THIS CODE ONLY REFERS TO MLP3X3, IT IS POSSIBLE TO MODIFY IT       #
#   FOR OTHER CLASSIFIERS BY JUST CHANGING path. ANYWAY, IT IS IMPORTANT      #
#   TO CHECK LABELS SECTION WHEN PLOTTING AS THEY ARE CLASSIFIER SPECIFIC     #
###############################################################################


import os 
import numpy as np
import rasterio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import rel_entr

#%%

lucas_stat_path = 'C:/Users/drikb/Desktop/Tirocinio/GRAPH NEURAL NETWORKS/Random forest project/Dati/LUCAS/Lucas statistics/'

lucas_stat = pd.read_csv(lucas_stat_path + 'lan_lcv_ovw_1_Data.csv', encoding='latin-1')

#%%

path = 'C:/Users/drikb/Desktop/Land Cover Classifier/Predictions/RF3x3/'

all_percentages = []
for trial in os.listdir(path):
    pred_path = path + trial + '/'
    percentages = []
    for img in os.listdir(pred_path):
        if img.endswith('.tif'):
            image = rasterio.open(pred_path + img)
            predictions = image.read(1)
            tot = len(np.where(predictions != 0)[0])
            image_percentages = []
            for category in np.unique(predictions)[1:]:
                num = len(np.where(predictions == category)[0])
                image_percentages.append(num/tot)
            percentages.append(image_percentages)
    all_percentages.append(percentages)
    
#%%

perc_array = np.array(all_percentages)

perc_array = np.mean(perc_array, axis = 1)

perc_mean = np.mean(perc_array, axis = 0)
perc_std = np.std(perc_array, axis = 0)
        

description = np.array(['Artificial land',
               'Cropland',
               'Woodland',
               'Shrubland',
               'Grassland',
               'Bareland',
               'Water',
               'Wetlands'])

overall_percentages = pd.DataFrame(np.array([description, perc_mean, perc_std]).T, columns = ['Land Cover', 'Value', 'std'])     


#%%

lazio_lucas_percentages = lucas_stat[lucas_stat['GEO'] == 'Lazio'][lucas_stat['UNIT'] == 'Percentage'][lucas_stat['TIME'] == 2018]

#%%

comparison_df = pd.DataFrame(np.array([overall_percentages['Land Cover'].values, 
                              overall_percentages['Value'].values.astype('float32'), 
                              overall_percentages['std'].values.astype('float32'),
                              lazio_lucas_percentages['Value'].values.astype('float32')/100]).T,
                             columns = ['Land Cover', 'Predictions', 'std', 'Truth'])


comparison_df['Difference'] = comparison_df['Truth'] - comparison_df['Predictions']
comparison_df['Relative Difference'] = (comparison_df['Truth'] - comparison_df['Predictions'])/comparison_df['Truth']

#%%


resh_comp = pd.concat([comparison_df.loc[:,('Land Cover', 'Predictions', 'std')].rename(columns = {'Predictions' : 'Value'}), 
                       comparison_df.loc[:,('Land Cover', 'Truth')].rename(columns = {'Truth' : 'Value'})], 
                      axis = 0,
                      keys = ['Predictions', 'Truth'],
                      names = ['Category', 'ID']).reset_index()

resh_comp['CV'] = resh_comp['std']/resh_comp['Value']



resh_comp.replace(resh_comp['std'][resh_comp['std'].isna()], 0)

#%%

###############################################################################
#                    DISTRIBUTION EVALUATION                                  #

differences_frame = pd.DataFrame(perc_array.T)
differences_frame = pd.concat([differences_frame, comparison_df[['Truth', 'Land Cover']]], axis = 1)


# MEAN DIFFERENCE # # # # # # 
each_difference = np.array([differences_frame['Truth'] - differences_frame.iloc[:,i] for i in range(10)], dtype = 'float32').T

mean_each_diff = np.mean(abs(each_difference), axis = 1)

std_each_diff = np.std(abs(each_difference), axis = 1)


mean_diff = np.mean(mean_each_diff)


# KULLBACK LEIBLER DIVERGENCE # # # # # # 

KL_each_div = np.array([sum(rel_entr(differences_frame['Truth'].values.tolist(), differences_frame.iloc[:,i].values.tolist())) for i in range(10)]).T

KL_overall = sum(rel_entr(comparison_df['Truth'].values.tolist(), comparison_df['Predictions'].values.tolist()))
    

#%%

display, (ax1, ax2, ax3) = plt.subplots(ncols = 3, figsize = (16, 6))
sns.set_theme(style="whitegrid")

# FIGURE 1 - comparison
fig1 = sns.barplot(data = resh_comp,
                x = 'Land Cover',
                y = 'Value',
                hue = 'Category',
                alpha = 0.9,
                palette = 'deep',
                edgecolor = 'black',
                ax = ax1) 


ax1.set_xticklabels(labels = fig1.get_xticklabels(), rotation = 30)
ax1.set_yticklabels(labels = [np.arange(0, 45, 5).astype('str')[i] + '%' for i in range(9)])


# FIGURE 2 - abs diff
fig2 = sns.barplot(data = comparison_df,
                x = 'Land Cover',
                y = 'Difference',
                alpha = 0.9,
                color = 'darkred',
                dodge = False,
                edgecolor = 'black',
                ax = ax2)
ax2.set_xticklabels(labels = fig2.get_xticklabels(), rotation = 30)
ax2.set_yticklabels(labels = [np.arange(-20, 30, 5).astype('str')[i] + '%' for i in range(9)])


# FIGURE 3 - rel diff
fig3 = sns.barplot(data = comparison_df,
                x = 'Land Cover',
                y = 'Relative Difference',
                alpha = 0.9,
                color = 'darkred',
                dodge = False,
                edgecolor = 'black',
                ax = ax3)
ax3.set_xticklabels(labels = fig3.get_xticklabels(), rotation = 30)
