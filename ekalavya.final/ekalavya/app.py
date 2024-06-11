from flask import Flask, render_template, request, jsonify
import cv2
from skimage.segmentation import slic
import numpy as np
import pandas as pd
import pickle
from markupsafe import Markup
from utils.fertilizer import fertilizer_dic
from utils.pest import disease_dic
from sklearn.ensemble import RandomForestClassifier


disease_varieties_description = {
    'Apple_scab': 'Apple___Apple_scab',
    'Black_rot': 'Apple___Black_rot',
    'Cedar_apple_rust': 'Apple___Cedar_apple_rust',
    'Powdery_mildew': 'Cherry_(including_sour)___Powdery_mildew',
    'Cercospora_leaf_spot_Gray_leaf_spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Common_rust': 'Corn_(maize)___Common_rust_',
    'Northern_Leaf_Blight': 'Corn_(maize)___Northern_Leaf_Blight',
    'g_Black_rot':'Grape___Black_rot',
    'Esca_Black_Measles':'Grape___Esca_(Black_Measles)',
    'Leaf_blight_Isariopsis_Leaf_Spot':'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Haunglongbing_Citrus_greening':'Orange___Haunglongbing_(Citrus_greening)',
    'Bacterial_spot':'Peach___Bacterial_spot',
    'bell___Bacterial_spot':'Pepper,_bell___Bacterial_spot',
    'Early_blight_p':'Potato___Early_blight',
    'Late_blight_p':'Potato___Late_blight',
    'Leaf_scorch':'Strawberry___Leaf_scorch',
    't_Bacterial_spot':'Tomato___Bacterial_spot',
    'Early_blight':'Tomato___Early_blight',
    'Late_blight':'Tomato___Late_blight',
    'Leaf_Mold':'Tomato___Leaf_Mold',
    'Septoria_leaf_spot':'Tomato___Septoria_leaf_spot',
    'Spider_mites_Two-spotted_spider_mite':'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Target_Spo':'Tomato___Target_Spo', 
    'Tomato_Yellow_Leaf_Curl_Virus':'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_mosaic_virus':'Tomato___Tomato_mosaic_virus',
    's_Powdery_mildew':'Squash___Powdery_mildew'
}

app = Flask(__name__)

# Load pre-trained machine learning models
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
model3 = pickle.load(open('model3.pkl', 'rb'))
model4 = pickle.load(open('model4.pkl', 'rb'))

with open("modelcrop.pkl", "rb") as f:
    model_crop = pickle.load(f)

with open("modelcrop1.pkl", "rb") as f:
    model_yield = pickle.load(f)

with open("modelcrop2.pkl", "rb") as f:
    model_fertilizer = pickle.load(f)

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

def predict_soil_attributes(image_path):
    blurred_image, segments = preprocess_image(image_path)
    rgb_values = extract_rgb_from_segments(blurred_image, segments)

    predicted_ph = model.predict(rgb_values.mean(axis=0).reshape(1, -1))[0]
    predicted_type = model1.predict(rgb_values.mean(axis=0).reshape(1, -1))[0]
    predicted_nitrogen = model2.predict(rgb_values.mean(axis=0).reshape(1, -1))[0]
    predicted_phosphorous = model3.predict(rgb_values.mean(axis=0).reshape(1, -1))[0]
    predicted_potassium = model4.predict(rgb_values.mean(axis=0).reshape(1, -1))[0]

    soil_data = {
        'pH': predicted_ph.tolist(),
        'Soil_Type': predicted_type,
        'Nitrogen': predicted_nitrogen.tolist(),
        'Phosphorous': predicted_phosphorous.tolist(),
        'Potassium': predicted_potassium.tolist()
    }

    return soil_data

def predict_top_crops(input_features, n=3):
    probabilities = model_crop.predict_proba([input_features])[0]  # Use predict_proba for classification
    classes = model_crop.classes_
    sorted_probs = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    top_n_crops = sorted_probs[:n]
    return top_n_crops

def calculate_total_yield(top_crops, area_under_cultivation, input_features):
    total_yield = {}
    for crop, probability in top_crops:
        expected_yield = model_yield.predict(input_features)[0]  # Assuming input_features contain required features
        total_yield[crop] = probability * area_under_cultivation * expected_yield
    return total_yield



#templates rendering
@app.route('/')
def open():
    return render_template('open.html')
@app.route('/enter', methods=['POST'])
def enter():
    return render_template('enter.html')
@app.route('/home', methods=['POST'])
def home():
    return render_template('home.html')
@app.route('/crop')
def crop():
    return render_template('yield.html')
@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')
@app.route('/pest')
def pest():
    return render_template('pest.html')

#results

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error='No file part')
    file = request.files['file']
    file_path = 'uploads/' + file.filename
    file.save(file_path)

    soil_data = predict_soil_attributes(file_path)

    return jsonify(soil_data)

@app.route('/result', methods=['POST'])
def result():
    input_features = [
       request.form.get('temperature', type=float),
        request.form.get('humidity', type=float),
        request.form.get('rainfall', type=float),
        request.form.get('soil_ph', type=float),
        request.form.get('nitrogen_content', type=float),
        request.form.get('phosphorus_content', type=float),
        request.form.get('potassium_content', type=float)
    ]
    input_features_reshape = np.array(input_features).reshape(1, -1)
    
    top_crops = predict_top_crops(input_features, n=3)
    area_under_cultivation = request.form.get('land_area', type=float)
    total_yield = calculate_total_yield(top_crops, area_under_cultivation, input_features_reshape)
    input_features1 = [
        request.form.get('soil_ph', type=float),
        request.form.get('nitrogen_content', type=float),
        request.form.get('phosphorus_content', type=float),
        request.form.get('potassium_content', type=float)
    ]
    input_features_reshape1 = np.array(input_features1).reshape(1, -1)
    fertilizer_used=model_fertilizer.predict(input_features_reshape1)
    return render_template('yield2.html', top_crops=top_crops, total_yield=total_yield,fertilizer_used=fertilizer_used )
@app.route('/fertilizer-predict', methods=['POST'])
def fertpredict():
    crop_name = str(request.form['cropname'])
    N = float(request.form['nitrogen_content'])
    P = float(request.form['phosphorus_content'])
    K = float(request.form['potassium_content'])

    df = pd.read_csv('fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer2.html', recommendation=response)  

@app.route('/pest-predict', methods=['POST'])
def pestpredict():
    disease_name = str(request.form['diseasename'])
    dic_disease_name=disease_varieties_description[disease_name]
    response1 = Markup(str(disease_dic[dic_disease_name]))
    return render_template('pest2.html', recommendation=response1)


if __name__ == '__main__':
    app.run(debug=True)
