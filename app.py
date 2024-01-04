from flask import Flask,render_template,request,jsonify,json
import yaml
import os
import joblib
import pandas as pd
import numpy as np

params_path='params.yaml'
webapp_root='docs'

static_dir=os.path.join(webapp_root,'static')
template_dir=webapp_root

app=Flask(__name__,static_folder=static_dir,template_folder=template_dir)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

def predict(data):
    config=read_params(params_path)
    model_dir_path=config['webapp_model_dir']
    model=joblib.load(model_dir_path)
    prediction=model.predict(data)
    return prediction[0]


       
           
            
         
     
    

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        try:
            if request.form:
                data_dict = {
                     'age': request.form['age'],
                     'bmi': request.form['bmi'],
                     'children': request.form['children'],
                     'sex': request.form['sex'],
                    'smoker': request.form['smoker'],
                    'region': request.form['region']
                }
                data=pd.DataFrame([list(data_dict.values())],
                                  columns=data_dict.keys())
                print(data)
                response=predict(data)
                return render_template("index.html",response=response)
        except Exception as e:
            print(e)
            error={'error':'Something went wrong!!.Try again'}
            return render_template("404.html",error=error)
    
    
    if request.is_json:
        try:
            data_dict = request.get_json()
            data = pd.DataFrame([data_dict])
            response=predict(data)
            result={"response":response}
            return result
        except Exception as e:
            print(e)
            error = {'error': 'Something went wrong!!.Try again'}
            return ({'error': (error)})
    
    else:
       return render_template("index.html") 

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)