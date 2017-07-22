
# coding: utf-8

# # Planet: Understanding the Amazon deforestation from Space challenge

# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
# 
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
# 
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **

# Start by adding the helper files to the python path

# In[1]:

import sys

sys.path.append('../src')
sys.path.append('../tests')


import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import data_helper
#from keras_helper import AmazonKerasClassifier
from src.xception_classifier import XceptionClassifier

#img_resize = (64, 64) # The resize size of each image
img_resize = (74, 74)  # nova velikost potrebna pro XCeption model
validation_split_size = 0.2
epochs = 20
batch_size = 128


train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
labels_df.head()


# Each image can be tagged with multiple tags, lets list all uniques tags

# In[5]:

# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))


# # Data preprocessing
# Preprocess the data in order to fit it into the Keras model.
# 
# Due to the hudge amount of memory the resulting matrices will take, the preprocessing will be splitted into several steps:
#     - Preprocess training data (images and labels) and train the neural net with it
#     - Delete the training data and call the gc to free up memory
#     - Preprocess the first testing set
#     - Predict the first testing set labels
#     - Delete the first testing set
#     - Preprocess the second testing set
#     - Predict the second testing set labels and append them to the first testing set
#     - Delete the second testing set

# In[9]:

x_train, y_train, y_map = data_helper.preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)
# Free up all available memory space after this heavy operation
gc.collect();


# In[10]:

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))
print(y_map)


# ## Create the neural network definition

# In[11]:

# classifier = AmazonKerasClassifier()
# classifier.add_conv_layer(img_resize)
# classifier.add_flatten_layer()
# classifier.add_ann_layer(len(y_map))

classifier = XceptionClassifier(image_size=img_resize[0])
classifier.add_classification(len(y_map))
train_losses, val_losses, train_acc, val_acc, fbeta_score = classifier.train_model(x_train, y_train, epochs, batch_size, validation_split_size=validation_split_size)


# ## Monitor the results

# Check that we do not overfit by plotting the losses of the train and validation sets

# In[12]:

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.savefig('Losses.png')

plt.clf()
plt.plot(train_acc, label='Training accuracy')
plt.plot(val_acc, label='Validation_accuracy')
plt.legend()
plt.savefig('Accuracy.png', transparent=False)


# Look at our fbeta_score

# In[13]:

print("fbeta_score je - ", fbeta_score)


# Before launching our predictions lets preprocess the test data and delete the old training data matrices

# In[14]:

del x_train, y_train
gc.collect()

x_test, x_test_filename = data_helper.preprocess_test_data(test_jpeg_dir, img_resize)
# Predict the labels of our x_test images
predictions = classifier.predict(x_test)


# Now lets launch the predictions on the additionnal dataset (updated on 05/05/2017 on Kaggle)

# In[15]:

del x_test
gc.collect()

x_test, x_test_filename_additional = data_helper.preprocess_test_data(test_jpeg_additional, img_resize)
new_predictions = classifier.predict(x_test)

del x_test
gc.collect()
predictions = np.vstack((predictions, new_predictions))
x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                                                                              x_test_filename.shape,
                                                                              predictions[0]))


# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class.
# 
# To do so we will take the median value of each classes.

# In[16]:

# For now we'll just put all thresholds to 0.2 
thresholds = [0.2] * len(labels_set)

# TODO complete
tags_pred = np.array(predictions).T
_, axs = plt.subplots(5, 4, figsize=(15, 20))
axs = axs.ravel()

for i, tag_vals in enumerate(tags_pred):
    sns.boxplot(tag_vals, orient='v', palette='Set2', ax=axs[i]).set_title(y_map[i])


# Now lets map our predictions to their tags and use the thresholds we just retrieved

# In[17]:

predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)


# Finally lets assemble and visualize our prediction for the test dataset

# In[18]:

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]


# In[19]:

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
print("final.df_head - ",final_df.head())


# In[20]:

tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');


# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...

# And save it to a submission file

# In[21]:

final_df.to_csv('../submission_file.csv', index=False)
classifier.close()


# That's it, we're done!
