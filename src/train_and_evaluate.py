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


def grid_search(x_train_trf,y_train):
     param_grid={
    'n_estimators':[100,120,140],
    'max_depth':[3,5,7,9],
    'min_samples_split':[5,10,12,15]}
     gsv=GridSearchCV(RandomForestRegressor(),param_grid,cv=5,scoring='r2',verbose=2,refit=True)
     gsv.fit(x_train_trf,y_train)
     best_model=gsv.best_estimator_
     return best_model




def train_and_evaluate(config_path):
    config=read_params(config_path)
    train_path=config['split_data']['train_path']
    test_path=config['split_data']['test_path']
    test_size=config['split_data']['test_size']
    random_state=config['base']['random_state']
    target_col=config['base']['target_col']
    
   

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
    
    
    x_train_trf=trf.fit_transform(x_train)
    best_model=grid_search(x_train_trf,y_train)
    
    
    pipeline=Pipeline([
        ('trf',trf),
        ('rf',best_model)
         #RandomForestRegressor(max_depth=3,min_samples_split=5,n_estimators=120))
    ])
    
    pipeline.fit(x_train,y_train)
    y_pred=pipeline.predict(x_test)
    return calucate_error(y_test,y_pred)

def calucate_error(y_test,y_pred):
    mean_abs_err=mae(y_test,y_pred)
    root_mean_squ_err=np.sqrt(mse(y_test,y_pred))
    r2_Score=r2(y_test,y_pred)
    print(f"mae: {mean_abs_err}\n",
          f"rmse: {root_mean_squ_err}\n",
          f"r2 socre: {r2_Score}")
   
    
    

    
    





if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_arg=arg.parse_args()
    print('****')
    train_and_evaluate(config_path=parsed_arg.config)
