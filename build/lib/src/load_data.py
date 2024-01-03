import os
import yaml
import pandas as pd
import argparse
from get_data import get_data,read_params



def load_data(config_path):
    config=read_params(config_path)
    raw_path=config['load_data']['raw_dataset_csv']
    load_data,new_cols=get_data(config_path)
    load_data.to_csv(raw_path,sep=',',index=False,header=new_cols)
    

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_arg=arg.parse_args()
    print('****')
    load_data(config_path=parsed_arg.config)

