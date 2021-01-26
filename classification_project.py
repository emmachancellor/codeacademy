import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np
# Loading the Data
# STEP 1: load in the data as a pandas DataFrame
data = pd.read_csv("heart_failure.csv")

# STEP 2: take a look at the columns and data types:
#print(data.info())

# STEP 3: print the death_event column
#print(Counter(data["death_event"]))

# STEP 4: extract the label column and assign to a new variable
y = data["death_event"]

# STEP 5: extract the features columns
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

# Data Preprocessing
# STEP 6: convert the categorical features in the DataFrame instance x to one-hot encoding vctors
x = pd.get_dummies(x)

# STEP 7: split the data
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# STEP 8: scale the numeric features
ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase', 'ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

# STEP 9: scale the training data
X_train = ct.fit_transform(X_train)

# STEP 10: scale the test data
X_test = ct.fit_transform(X_test)

#Prepare Labels for Classification
# STEP 11: initialize a label encoder
le = LabelEncoder()

# STEP 12: fit the encoder to the training labels and convert the training labels according to the trained encoder
Y_train = le.fit_transform(Y_train.astype(str))

# STEP 13: convert test labels
Y_test = le.fit_transform(Y_test.astype(str))

# STEP 14/15: transform the encoded training labels into a binary vector 
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Design the model
# STEP 16: initialize the model
model = Sequential()

# STEP 17: create an input layer
model.add(InputLayer(input_shape = (X_train.shape[1],)))

# STEP 18: create a hidden layer
model.add(Dense(12, activation = "relu"))

# STEP 19: create an outpul layer
model.add(Dense(2, activation = "softmax"))

# STEP 20: compile the model
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])

# Train and evaluate the model
# STEP 21: fit the model
model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose = 1)

# STEP 22: evaluate the model
loss, acc = model.evaluate(X_test, Y_test, verbose = 0)
print("Loss:", loss, "\n Accuracy:", acc)

# Generating a classification report
# STEP 23: get the predictions for the test data with the trained model
y_estimate = model.predict(X_test, verbose = 0)

# STEP 24: select the indices of tre classes for each label encoding in y_estimate
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(Y_test, axis = 1)
print(classification_report(y_true, y_estimate))
