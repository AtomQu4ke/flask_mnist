from flask import Flask, render_template, request
global graph,model
from funk import reshape_img
import pickle
import tensorflow as tf
graph = tf.get_default_graph()

app = Flask(__name__)
model = pickle.load(open('C:/Users/marie/OneDrive/Documents/formation-simplon/Deep_learning/MNIST/model_mnist2','rb'))

@app.route('/')
@app.route('/home')
def view():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_img():
    if request.method == 'POST':
        f = request.files['file']
        name = f.filename
        mon_image = reshape_img(f,name)
        with graph.as_default():
            prediction = ("Je crois que c'est un "+str(model.predict_classes(mon_image)[0]),name)
    return render_template('predict.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True)
