# LUCAS POINTS AND SENTINEL-2 IMAGES FOR LAND COVER CLASSIFICATION

Federico Pavesi - [LinkedIn](https://www.linkedin.com/in/federico-pavesi-b2360323a/)

Visualise results at [this link](https://federicopavesi-github-app-app-tn5gy3.streamlitapp.com/)

![RGB Lazio image](https://github.com/FedericoPavesi/Lucas_points_for_Sentinel2_LandCover_Download/blob/main/RGB.PNG) | ![Lazio land cover map](https://github.com/FedericoPavesi/Lucas_points_for_Sentinel2_LandCover_Download/blob/main/CLASSIFIED.PNG)

__In this work__, what it is done is to propose a methodology which might overcome issues experimented in literature in the task of __building an effective land-cover classifier__. It starts by creating a balanced database composed by pixel (and neighbourhood) reflectance, coming from high precision remote sensing data with an associated land cover class. Then, a machine learning (random forest) and a deep learning (MLP) algorithms are trained and tuned until the best architecture is found. The result of this process is four trained models: two for when pixel neighbourhood is taken into account and two for when it is not. These models are compared between them in terms of test accuracy and training speed. After this comparison, a map of region Lazio is created using pixels median reflectance over 2018. This map is fed to each classification algorithm to finally produce respective land cover masks. The final step is to compute masks summary statistics (percentages of land cover classes coverage) and compare them with Lucas regional statistics in order to evaluate the precision of each mask.

__This repository__ contains all necessary codes to build the training dataset, train classifiers, evaluate algorithms variability, download regional reflectance (by default Italian region Lazio) and perform map predictions. 

__Each folder__ represents a step in the process, and contains all the codes needed to perform it. Attached, it is possible to find a .README with an explanation of each passage and additional requirements.

__Codes__ are in python 3.9 language (some steps in .py and some in .ipynb) and were built in two different anaconda environments. Specifically, an [Earth Engine](https://earthengine.google.com/) and a [Tensorflow](https://www.tensorflow.org/) environment are needed for correctly run each procedure (I do not exclude it is possible to use a singe environment with Earth Engine and Tensorflow but I strongly suggest to keep them separately), follow installation procedures available from their respective website to correctly build both environments. Required libraries are mentioned step by step.

__Author's note__: Proposed procedure is very simple and it's only the incipit of a wider ongoing research. Following this workflow it is very unlikely one could end up with an effective land cover classifier which is able to produce reliable soil classification. Nevertheless, it could provide useful baselines for future research from an analysis of both its strenghs and weaknesses. 

