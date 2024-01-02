import os
import yaml
import pandas as pd
import argparse
from split_data import split_data
from get_data import read_params,get_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as r2,mean_absolute_error as mae,mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor

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
    print(x_train.head(),x_train.shape,y_train.head(),y_train.shape)
    print(x_test.head(),x_test.shape)



if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_arg=arg.parse_args()
    print('****')
    train_and_evaluate(config_path=parsed_arg.config)