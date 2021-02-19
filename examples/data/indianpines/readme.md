# Indian Pines data set
[http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes]

The data sets contain 10249 pixels of hyperspectral images of a given landscene in Indiana. Each pixel contains 220 values representing 220 different hyperspectral bands of that same pixel in the scene. The goal is to classify each pixel into one out of 16 classes (labels), being the soil cover type (corn, gras, ...).

The original landscene images are of dimension 145x145 pixels. From this total of 21025 pixels, those that belong to none of the 16 classes had been removed, leaving the 10249 selected ("color") pixels for training and validation. The index map idmap_color_to_global.dat identifies the position of those color pixels in the overall image. Here, reshape from vector to matrix 145x145 is columnswise. 

* features_All.dat, labels_All.dat contain all of the pixels (sorted!): 10249 rows (pixels), with 220 columns (hyperspectral bands) in the features and 16 columns in the labels (land cover type)
* features_*_shuffle.dat, labels_*_shuffle.dat contain randomly selected pixels and there corresponding labels, 4000 in the training, 2000 in the validation data files. 
* idmap_*_to_color.dat maps the training and validation pixels back to their location inside *_All.dat.  


