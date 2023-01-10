import pickle
from flask import Flask , render_template , request
import numpy as np
# Create the object of Flask class
app=Flask(__name__)
# to open file in read mode
file = open("scale.pkl","rb")
file1 = open("model.pkl","rb")



# to read data from file
scale=pickle.load(file)
model=pickle.load(file1)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def pred():
    features=[float(x) for x in request.form.values()]
    # to converts all input features in numpy 2D array
    final=[np.array(features)]
    # apply scaling on final input
    final=scale.transform(final)
    # to predict the model
    pred=model.predict(final)[0]

    if pred==0:
        pred='Iris-setosa'
    elif pred==1:
        pred='Iris-versicolor'
    else:
        pred='Iris-virginica'

    return render_template('result.html',pred=pred)

    
# Main program
app.run(debug=True,use_reloader=False)
