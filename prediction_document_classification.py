# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 22:55:11 2018

@author: varsha.vishwakarma
"""
from sklearn.model_selection import train_test_split
from data_helpers import load_data
import numpy as np
from keras.models import load_model
from numpy import genfromtxt
x_test = genfromtxt(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\work-day-classification\test_input_dataset.csv', delimiter=',')
y_test = genfromtxt(r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\work-day-classification\test_output_dataset.csv', delimiter=',')

model_file = r'C:\Users\varsha.vishwakarma\Documents\CNN-text-classification\work-day-classification\weights.002-0.9767.hdf5'

# model loading
model = load_model(model_file)
print('model loaded')
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)
y_Test = y_test[:,1:2]
y_Pred = y_pred[:,1:2]

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_Test, y_Pred)

