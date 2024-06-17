from flask import Flask, render_template, request, jsonify
import joblib
import csv

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
        symptom_data = request.json
        symptoms = symptom_data.get('symptoms', [])

        symptom_weights = {}

        with open('symptom-severity.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                symptom_weights[row['Symptom']] = float(row['Weight'])

        weights = []
        for symptom in symptoms:
            if symptom in symptom_weights:
                weights.append(symptom_weights[symptom])
            else:
                weights.append(0)  # Default weight or handle differently as per your model requirements

        # Ensure input_data is a list of lists (2D array) if needed by your model
        input_data = [weights]
        prediction = model.predict(input_data)[0]
        response_message = f"The predicted disease is: Success"
        return jsonify(response_message)
    
@app.route('/diseasePredict')
def diseasePredict(disease):
    with open('symptom-severity.csv', 'r') as file:
        reader = csv.DictReader(file)     
    discription = {}
    for row in reader:
        discription[row['Disease']] = row['Description']
    
    discrip = discription(disease)

    return jsonify({'message': discrip})


if __name__ == "__main__":
    app.run(debug=True)
