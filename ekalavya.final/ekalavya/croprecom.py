import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pickle

data = pd.read_csv("crop_recommendation.csv")

# Dictionary containing the yield ranges for each crop
crop_yield_ranges = {
    'rice': (3000, 4000),
    'maize': (5000, 10000),
    'chickpea': (800, 1200),
    'kidney beans': (1500, 2000),
    'pigeon peas': (800, 1200),
    'moth beans': (500, 800),
    'mung beans': (500, 800),
    'black gram': (500, 800),
    'lentil': (800, 1200),
    'pomegranate': (30000, 50000),
    'banana': (30000, 50000),
    'mango': (30000, 50000),
    'grapes': (30000, 50000),
    'watermelon': (10000, 20000),
    'muskmelon': (10000, 20000),
    'apple': (10000, 20000),
    'orange': (10000, 20000),
    'papaya': (10000, 20000),
    'coconut': (50000, 80000),
    'cotton': (1000, 2000),
    'jute': (500, 1000),
    'coffee': (1000, 2000),
}

def generate_yield(row):
    crop = row['label']
    yield_range = crop_yield_ranges.get(crop, (0, 0))  # Default range if crop not found
    return np.random.randint(yield_range[0], yield_range[1] + 1)

# Apply the generate_yield function to each row to fill the 'Yield' column
data['Yield'] = data.apply(generate_yield, axis=1)

# Save the updated dataset to a new CSV file
data.to_csv('crop_recommendation1.csv', index=False)

print("Dataset with filled Yield field saved successfully.")



data = pd.read_csv("crop_recommendation1.csv")


data2 = pd.read_csv("Fertilizer Prediction.csv")


x = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']


y1= data['Yield']


x2 = data2[['Nitrogen','Potassium','Phosphorous','Temparature',]]
y2 = data2['Fertilizer Name']


model = RandomForestClassifier(n_estimators = 25, random_state=2)
model.fit(x, y)

model1 = RandomForestRegressor(n_estimators = 25, random_state=2)
model1.fit(x, y1)

model2 = RandomForestClassifier(n_estimators = 25, random_state=2)
model2.fit(x2, y2)

def predict_top_crops(input_features, n=3):
    probabilities = model.predict_proba([input_features])[0]
    classes = model.classes_
    sorted_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    top_n_crops = sorted_probs[:n]
    return top_n_crops

# input_features = [50,40 ,36, 20.8, 82.0, 12.1, 300.9]  
# input_features1 = [50,36,40,20.8]

# top_3_crops = predict_top_crops(input_features)
# print("Top 3 probable crops with their probabilities:")
# for crop, probability in top_3_crops:
#     print(f"{crop}: {probability * 100:.2f}%")
# input_features_array = np.array(input_features)
# input_features = input_features_array.reshape(1, -1)
# predicted_yield = model1.predict(input_features)
# print("Predicted Yield:", predicted_yield, "kg/acre")

# input_features_array1 = np.array(input_features1)
# input_features1 = input_features_array1.reshape(1, -1)
# predicted_fertilizer = model2.predict(input_features1)
# print("Predicted fertilizer :", predicted_fertilizer)

Pkl_Filename = "modelcrop.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)

Pkl_Filename = "modelcrop1.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model1, file)

Pkl_Filename = "modelcrop2.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model2, file)