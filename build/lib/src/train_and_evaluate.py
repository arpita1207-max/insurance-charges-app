import os
import yaml
import pandas as pd
import numpy as np
import argparse
from split_data import split_data
from get_data import read_params,get_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2,mean_absolute_error as mae,mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import json


def grid_search(x_train_trf,y_train):
    #param grid to pass to grid search
     param_grid={
    'n_estimators':[100,120,140],
    'max_depth':[3,5,7,9],
    'min_samples_split':[5,10,12,15]}
     # fit in the grid search cv
     gsv=GridSearchCV(RandomForestRegressor(),param_grid,cv=5,scoring='r2',verbose=2,refit=True)
     gsv.fit(x_train_trf,y_train)
     best_model=gsv.best_estimator_
     best_params=gsv.best_params_
     return best_model,best_params

def calucate_error(y_test,y_pred):
    mean_abs_err=mae(y_test,y_pred)
    root_mean_squ_err=np.sqrt(mse(y_test,y_pred))
    r2_Score=r2(y_test,y_pred)
    return mean_abs_err,root_mean_squ_err,r2_Score


def train_and_evaluate(config_path):
    config=read_params(config_path)
    train_path=config['split_data']['train_path']
    test_path=config['split_data']['test_path']
    test_size=config['split_data']['test_size']
    random_state=config['base']['random_state']
    target_col=config['base']['target_col']
    model_dir=config['model_dir']
    
   

    train=pd.read_csv(train_path)
    test=pd.read_csv(test_path)
    x_train,x_test,y_train,y_test=train_test_split(train.drop(columns=target_col),
                                                   train[target_col],
                                                   test_size=test_size,
                                                   random_state=random_state)
    


    cat_cols=x_train.select_dtypes(include='object').columns
    num_cols=x_train.select_dtypes(exclude='object').columns
    
    trf=ColumnTransformer([('encoder',OneHotEncoder(drop='first'),cat_cols),
                       ('scaler',StandardScaler(),num_cols)
                       ],remainder='passthrough',verbose_feature_names_out=False
                     )
    
    
    #x_train_trf=trf.fit_transform(x_train)
    #best_model,best_params=grid_search(x_train_trf,y_train)
    
    max_depth=config['estimators']['rfr']['params']['max_depth']
    min_samples_split=config['estimators']['rfr']['params']['min_samples_split']
    n_estimators=config['estimators']['rfr']['params']['n_estimators']
    
    pipeline=Pipeline([
        ('trf',trf),
        ('rf',
         RandomForestRegressor(max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               n_estimators=n_estimators))
    ])
    
    pipeline.fit(x_train,y_train)
    y_pred=pipeline.predict(x_test)
    (mean_abs_err,root_mean_squ_err,r2_Score)=calucate_error(y_test,y_pred)
    print(f"mae: {mean_abs_err}\n",
          f"rmse: {root_mean_squ_err}\n",
          f"r2 socre: {r2_Score}")
    
    scores_file=config['reports']['scores']
    params_file=config['reports']['params']
    with open(scores_file,'w') as f:
        scores={
            "mae":mean_abs_err,
            "rmse":root_mean_squ_err,
            "r2 score": r2_Score
        }
        json.dump(scores,f)
        
    with open(params_file,'w') as f:
        params={
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'n_estimators':n_estimators
        }
        json.dump(params,f)
    
    
   
    
    os.makedirs(model_dir,exist_ok=True)
    model_path=os.path.join(model_dir,"model.joblib")
    joblib.dump(pipeline[-1],model_path)

    
if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_arg=arg.parse_args()
    print('****')
    train_and_evaluate(config_path=parsed_arg.config)
