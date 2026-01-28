import pandas as pd
from flask import Flask, render_template,request
import pickle


app = Flask(__name__)
data=pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route("/")
def home():
    locations=sorted(data["location"].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    input_df = pd.DataFrame([{
        'location': location,
        'total_sqft': sqft,
        'bath': bath,
        'bhk': bhk
    }])

    prediction = pipe.predict(input_df)[0]

    return f'Estimated Price: ₹ {round(prediction,2)} Lakhs'

if __name__ == "__main__":
    app.run(debug=True,port=5001)
