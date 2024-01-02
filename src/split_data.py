import os
import yaml
import pandas as pd
import argparse
from load_data import load_data
from get_data import read_params,get_data

def split_data(config_path):
    config=read_params(config_path)
    df,new_cols=get_data(config_path)
    train_path=config['split_data']['train_path']
    test_path=config['split_data']['test_path']
    train=df.iloc[0:1004]
    test=df.iloc[1004:]
    train.to_csv(train_path,sep=',',header=new_cols,index=False)
    test.to_csv(test_path,sep=',',header=new_cols,index=False)
    
    

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_arg=arg.parse_args()
    print('****')
    split_data(config_path=parsed_arg.config)
