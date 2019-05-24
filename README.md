# railroad_classifier


#### 1.	Sequential_Model_01.py 
modified CNN model to predict new images’ class. (You don’t have to download or run it.)
 
#### 2.	Sequential_categorical.h5  
The best model saved from 50 iteration of model fit. This model showed 99% test accuracy 

#### 3.	Railroad_Classifier.py
python codes for separate images to ‘railroad’ and ‘other’ classes.

<br>
<br>
To separate collected images automatically into two different folder (e.g. “/railroad” and “/other”), download ‘Sequential_categorical.h5 ‘ in the python working directory, and use **railroad_classifier(dir_Dataset, dir_Railroad, dir_Other, filetype='png')** function in  ‘Railroad_Classifier.py’
