from flask import Flask, render_template, jsonify, request, url_for 
import src.model as model

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    pred = model.predict_image(request.files['image'], request.files['image'].filename)
    return render_template('display.html',result=pred)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000')