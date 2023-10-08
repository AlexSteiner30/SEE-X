from flask import Flask, render_template, request
from fileinput import filename 
from predict import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/start')
def start():
    return render_template('start.html')

@app.route('/about-us')
def about():
    return render_template('about-us.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/outcome', methods=['POST'])
def outcome():
    f = request.files['file'] 
    f.save('static/files/' + f.filename)

    result = predict('static/files/' + f.filename)

    output = "has been diagnosed with brain aneurysm"

    if result == 0:
        output = "doesn't have brain aneurysm"

    # 1 = yes
    # 0 = no
    
    return render_template('outcome.html', output=output)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
