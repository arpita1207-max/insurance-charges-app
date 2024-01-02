## read paramters
## process
## return dataframe
import os
import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config=read_params(config_path)
    #print(config)
    data_path=config["data_source"]["s3_source"]
    df=pd.read_csv(data_path,sep=',',encoding='utf-8')
    #print(df.head())
    print('***')
    return df,df.columns




if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--config",default="params.yaml")
    parsed_arg=arg.parse_args()
    print('****')
    load_data,new_cols=get_data(config_path=parsed_arg.config)
    print(load_data,new_cols)
