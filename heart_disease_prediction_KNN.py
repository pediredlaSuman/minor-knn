# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 16:11:42 2023

@author: Suman
"""

import numpy as np
import pickle 
# loading the saved model
loaded_model = pickle.load(open("C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_heart_sav_svm", 'rb'))
input_data = (37,1,2,130,250,0,1,187,0,3.5,0,0,2)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person  does not has heart_attack')
else:
  print('The person has heart_attack')

