from flask import Flask,request, render_template
import numpy as np
import pandas as pd
import pickle
import sklearn
print(sklearn.__version__)
#loading models
dtr = pickle.load(open('dtr.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
df = pd.read_csv("./yield_df.csv")
countries =  df['Area'].unique()
crops = df['Item'].unique()

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', countries = countries, crops=crops, prediction=False)


@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        avg_rain = request.form['avg_rain']
        fertlizer = request.form['fertlizer']
        avg_temp = request.form['avg_temp']
        Area = 'India'
        Item  = request.form['Item']

        features = np.array([[Year,avg_rain,fertlizer,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction, countries=countries, crops=crops)

if __name__=="__main__":
    app.run(debug=True)