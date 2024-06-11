import pandas as pd
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from skimage.segmentation import slic 
import warnings
import pickle


soil_dataset = pd.read_csv('soilpH_rgb.csv')


x = soil_dataset[['R', 'G', 'B']]
y = soil_dataset['pH']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y1=soil_dataset['Soil_Type']


x_train, x_test, y1_train, y1_test = train_test_split(x, y1, test_size=0.2, random_state=42)

model1 = RandomForestClassifier(n_estimators=100, random_state=42)
model1.fit(x_train, y1_train)

y2=soil_dataset['Nitrogen']


x_train, x_test, y2_train, y2_test = train_test_split(x, y2, test_size=0.2, random_state=42)

model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(x_train, y2_train)

y3=soil_dataset['Phosphorous']

x_train, x_test, y3_train, y3_test = train_test_split(x, y3, test_size=0.2, random_state=42)

model3 = RandomForestRegressor(n_estimators=100, random_state=42)
model3.fit(x_train, y3_train)

y4=soil_dataset['Potassium']


x_train, x_test, y4_train, y4_test = train_test_split(x, y4, test_size=0.2, random_state=42)

model4 = RandomForestRegressor(n_estimators=100, random_state=42)
model4.fit(x_train, y4_train)

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    segments = slic(blurred_image, n_segments=100, compactness=10)
    return blurred_image, segments


def extract_rgb_from_segments(image, segments):
 
    rgb_values = []
    for segment_id in np.unique(segments):
        mask = (segments == segment_id)
        masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
        average_rgb = np.mean(masked_image[mask], axis=0)
        rgb_values.append(average_rgb)
    return np.array(rgb_values)

def predict_ph_from_image(image_path, model):
   
    blurred_image, segments = preprocess_image(image_path)
    rgb_values = extract_rgb_from_segments(blurred_image, segments)
    predicted_ph = model.predict(rgb_values)
    return np.mean(predicted_ph)

def predict_type_from_image(image_path, model):
   
    blurred_image, segments = preprocess_image(image_path)
    rgb_values = extract_rgb_from_segments(blurred_image, segments)
    mean_rgb = np.mean(rgb_values, axis=0)
    mean_rgb_2d = mean_rgb.reshape(1, -1)
    predicted_type = model1.predict(mean_rgb_2d)
    return predicted_type[0]

def predict_Nitrogen_from_image(image_path, model):
   
    blurred_image, segments = preprocess_image(image_path)
    rgb_values = extract_rgb_from_segments(blurred_image, segments)
    predicted_Nitrogen = model2.predict(rgb_values)
    return np.mean(predicted_Nitrogen)

def predict_Phosphorous_from_image(image_path, model):
   
    blurred_image, segments = preprocess_image(image_path)
    rgb_values = extract_rgb_from_segments(blurred_image, segments)
    predicted_Phosphorous = model3.predict(rgb_values)
    return np.mean(predicted_Phosphorous)

def predict_Potassium_from_image(image_path, model):
   
    blurred_image, segments = preprocess_image(image_path)
    rgb_values = extract_rgb_from_segments(blurred_image, segments)
    predicted_Potassium = model4.predict(rgb_values)
    return np.mean(predicted_Potassium)

# image_path = 'C:\\Users\\dell\\OneDrive\\Desktop\\cultivodup\\Black_10.jpg'
# predicted_ph = predict_ph_from_image(image_path, model)
# predicted_type=predict_type_from_image(image_path, model)
# predicted_Nitrogen=predict_Nitrogen_from_image(image_path, model)
# predicted_Phosphorous=predict_Phosphorous_from_image(image_path, model)
# predicted_Potassium=predict_Potassium_from_image(image_path, model)
# print("Predicted pH:", predicted_ph)
# print("Predicted Type:", predicted_type)
# print("Predicted Nitrogen:", predicted_Nitrogen)
# print("Predicted Phosphorous:", predicted_Phosphorous)
# print("Predicted Potassium:", predicted_Potassium)

Pkl_Filename = "model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
Pkl_Filename = "model1.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model1, file)
Pkl_Filename = "model2.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model2, file)
Pkl_Filename = "model3.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model3, file)
Pkl_Filename = "model4.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model4, file)
