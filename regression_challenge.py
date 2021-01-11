import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score

dataset = pd.read_csv("admissions_data.csv")
#print(data_set.describe())
#print(data_set.columns)

dataset = dataset.drop(["Serial No."], axis = 1)
labels = dataset.iloc[:,-1]
features = dataset.iloc[:, 0:-1]

# split data into testing and training data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 5)

# Normalize the data

normalizer = Normalizer()

scaled_feats_train = normalizer.fit_transform(features_train)

scaled_feats_test = normalizer.transform(features_test)

features_train_scaled = pd.DataFrame(scaled_feats_train)
features_test_scaled = pd.DataFrame(scaled_feats_test)

def design_model(X, learning_rate):
  # Create the neural network
  my_model = Sequential()
  input = InputLayer(input_shape = features.shape[1],)
  my_model.add(input)
  my_model.add(Dense(16, activation = "relu"))
  my_model.add(Dense(1))
  # Optimizers
  opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
  # Compile
  my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)
  return my_model

def fit_model(f_train, l_train, learn_rate, num_epochs):
  model = design_model(features_train, learning_rate)
  es = EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 20)
  admit = model.fit(features_train, labels_train, epochs = num_epochs, batch_size = 16, verbose = 0, validation_split = 0.2, callbacks = [es])
  return admit

learning_rate = 0.1
num_epochs = 500
admit = fit_model(features_train_scaled, labels_train, learning_rate, num_epochs)

res_mse, res_mae = admit.evaluate(features_test, labels_test, verbose = 0)
# Do extensions code below

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(history.history['mae'])
ax1.plot(history.history['val_mae'])
ax1.set_title('model mae')
ax1.set_ylabel('MAE')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
 
  # Plot loss and val_loss over each epoch
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')
 
# used to keep plots from overlapping each other  
fig.tight_layout()
fig.savefig('static/images/my_plots.png')
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below

# fig.savefig('static/images/my_plots.png')
