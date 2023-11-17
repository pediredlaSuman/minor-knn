
import numpy as np
import pickle 
# loading the saved model
loaded_model = pickle.load(open("C:/Users/Suman/OneDrive/Desktop/Minor/trained_model_Parkinson_sav_svm",'rb'))
input_data=(119.992,157.302,74.997,0.0037,0.0054,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person does not has Parkinson')
else:
  print('The person has Pakinson')
