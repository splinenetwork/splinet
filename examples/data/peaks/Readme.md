# Peaks example

Supervised classification problem to classify grid points in [-3,3]^2 (features) to one out of five classes (labels). Classes correspond to five level sets of some previously generated function, visualized by color in figure levelsets.png. The data set contains a total of 5000 grid point coordinates with corresponding class probability vectors, stored row-wise:

* Rows in features.dat: Grid point coordinates (x,y) in [-3,3]^2
* Rows in labels.dat: Unit vectors in R^5 containing the probabilities of (x,y) belonging to the five classes, e.g. the vector (0, 0, 1, 0, 0) means that this row's coordinate belongs to the third class. 

The data is normalized in the sense that it contains 1000 points per class. Further, the data is sorted along the classes, i.e. the first 1000 rows are class 1, the next 2000 rows are class 2, etc, it is therefore necessary to draw randomly (uniform) from the data during training.


