from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
vector = pickle.load(open('vector.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')

@app.route('/prediction', methods =['POST'])
def home():

    
    text = request.form['age']
    
    
    arr = [text]
    print(arr)
    # arr[0][1] =(le.transform([[arr[0][1]]]))[0]
    arr = vector.transform(arr)
   
    

    



    pred = model.predict(arr)
    proba = round(model.predict_proba(arr)[0][1]*100,2)
    return render_template('after.html', pred =pred ,proba = proba, arr=arr)
    

if __name__ == "__main__":
    app.run(debug=True)