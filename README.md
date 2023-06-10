#### AGE_DETECTION_MODEL :

A deep convolutional neural network (CNN) to predict the age of a person based on their facial attributes. Approach involves converting the problem into a multi-class classification task with classes being Young, Middle and Old.

A training accuracy of 85.24% and testing accuracy of 78.15% suggests that your model is performing reasonably well on the data. However, it would be helpful to evaluate the model's performance further using other metrics such as precision, recall, and F1 score.


--->>>>>>>>>>>.     Model Design : 

Importing Libraries: The necessary libraries and modules are 
imported at the beginning of the code. These include libraries for data 
handling (numpy, pandas), visualization (matplotlib, seaborn), image 
processing (skimage), and machine learning (keras).

• Configuration and Visualization Setup: The code includes 
configurations and settings for visualizations using seaborn and 
matplotlib. It sets the style, color palette, and figure size for the plots.

• Data Loading: The code loads the dataset from CSV files using the 
pd.read_csv() function. It reads the training data into the train DataFrame 
and the testing data into the test DataFrame.

• Data Exploration and Visualization: The code includes functions 
for data exploration and visualization. The bar_plot() function generates 
a bar plot to visualize the distribution of data points across different 
classes. It provides insights into the class imbalance and helps 
understand the dataset.

• Image Processing: The code includes functions for image processing. 
The show_random_image() function selects a random image from the 
dataset and displays it along with its corresponding age class. The 
resize_img() function resizes the images to a specified dimension (32x32 
in this case) using the skimage library's resize() function. The images are 
also converted to float32 data type and normalized by dividing by 255.

• Data Preprocessing: The code preprocesses the data to prepare it for 
training the age prediction model. It uses label encoding from the scikitlearn library to convert the age class labels into numeric values. It also 
performs one-hot encoding using keras.utils.np_utils.to_categorical() to 
convert the labels into a binary matrix representation.

• Model Architecture: The code defines the architecture of the age 
prediction model using the Keras Sequential API. It adds various layers 
to the model, including convolutional layers, pooling layers, dropout 
layers, and dense layers. The model architecture is designed to extract 
features from the input images and learn a mapping between the input 
images and age classes.

• Model Compilation and Training: The code compiles the model by 
specifying the optimizer, loss function, and evaluation metrics using the 
compile() function. It then trains the model on the preprocessed training 
data using the fit() function. The training process involves iterating over 
the data in batches for a specified number of epochs. The training 
progress and performance metrics are displayed during the training 
process.

• Model Evaluation and Prediction: The code uses the trained model 
to make predictions on the training data (pred_train) and the testing data 
(pred). It also performs inverse label encoding to convert the predicted 
numeric labels back to the original age class labels. Finally, the predicted 
age classes are added to the test DataFrame and saved to a CSV file.








-->>>>>>>>>>>>>>>>>>>>>>>> Description : 

The code you provided uses the following libraries and implements the age detection 
algorithm:

Libraries:

• numpy: A library for numerical computing with arrays, used for handling numerical operations 
on data.

• pandas: A library for data manipulation and analysis, used for reading and manipulating 
structured data in tabular form.

• matplotlib: A plotting library for creating visualizations, used for generating various types of 
plots and charts.

• seaborn: A data visualization library based on matplotlib, used for creating aesthetically pleasing 
statistical graphics.

• skimage.transform: A module from the scikit-image library, used for image resizing and 
transformation operations.

• os: A module for interacting with the operating system, used for file and directory manipulation.

• random: A module for generating random numbers and making random selections, used for 
selecting random images from the dataset.

• keras: A high-level neural networks API, used for building and training deep learning models.

• LabelEncoder from sklearn.preprocessing: A utility class from scikit-learn for encoding 
categorical labels as integer values.

• Sequential, Dense, Flatten, InputLayer, Conv2D, MaxPooling2D, and Dropout from keras.layers: 
Classes for building different layers of a neural network model.

--------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Algorithm:

• Data Loading: The code loads the training and test datasets using pd.read_csv from the train.csv 
and test.csv files, respectively.

• Data Visualization: The bar_plot function is defined to create a bar plot showing the distribution 
of classes in the training dataset.

• Random Image Display: The show_random_image function selects a random image from the 
training dataset and displays it along with its age class.

• Image Resizing: The resize_img function resizes all the images in the training and test datasets to 
a fixed size of 32x32 pixels using the resize function from skimage.transform.

• Data Preprocessing: The image data is normalized by dividing the pixel values by 255, converting 
them to the range of 0 to 1.

• Bar Plot: The bar_plot function is called to create a bar plot showing the distribution of age 
classes in the training dataset.

• Label Encoding: The age class labels in the training dataset are encoded using LabelEncoder 
from sklearn.preprocessing, and then one-hot encoded using keras.utils.np_utils.to_categorical.

• Model Architecture: A convolutional neural network (CNN) model is defined using Sequential 
from keras.models and various layers such as Conv2D, MaxPooling2D, Flatten, Dense, and 
Dropout from keras.layers. The model architecture includes multiple convolutional layers, 
pooling layers, and fully connected layers.

• Model Compilation and Training: The model is compiled with an optimizer (Adam) and a loss 
function (categorical cross-entropy). It is then trained on the resized training images (train_x) 
and their corresponding labels (train_y) using model.fit.

• Prediction: The trained model is used to make predictions on both the training and test images 
(pred_train and pred, respectively). The predictions are then transformed back to their original 
class labels using lb.inverse_transform.

• Submission: The predicted age classes for the test dataset are added to the test DataFrame, and 
the DataFrame is saved as a CSV file named submission.csv.

• Random Image Prediction: A random image from the training dataset is displayed along with its 
original and predicted age classes
