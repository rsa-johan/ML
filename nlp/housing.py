import os
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request as req
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
from pandas.plotting import scatter_matrix
import sklearn.model_selection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def fetch_data(dir_url: str, dir_path: str, file_name: str):
    file_path = Path(dir_path) / file_name
    if file_path.is_file():
        print("File path exists")
        return
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    req.urlretrieve(dir_url, file_path)
    with tarfile.open(file_path) as tarball:
        tarball.extractall(path=dir_path)

def read_data(dir_path: str, extracted_dir_path: str, file_path: str):
    return pd.read_csv(Path(dir_path)/extracted_dir_path/file_path)

def get_null_count_cls(data: pd.DataFrame):
    isnulls = data.isnull().sum()
    return isnulls[isnulls > 0]

def get_val_count(data: pd.DataFrame, column: str):
    return data[column].value_counts()

def s_splitter(data: pd.DataFrame, stratified_col: str, t_size=0.2, random_state=42):
    split = train_test_split(data, test_size=t_size, random_state=random_state, stratify=data[stratified_col])
    return split

def drop_col(col_name: str, *data: pd.DataFrame):
    dropped_data = []
    for i in range(len(data)):
        print(i)
        dropped_data.append(data[i].drop(columns=[col_name]))
    return tuple(dropped_data)

def fill_na_strategy(data: pd.DataFrame, fill_strat="mean", remove_cols: list[str]=[]):
    data_rep = data
    if len(remove_cols) > 0:
        data_rep = data.drop(columns=remove_cols)

    data_rep = data_rep.select_dtypes(include=[np.number])
    print(data_rep.head())
    strat_imputer = SimpleImputer(strategy=fill_strat)
    strat_imputer.fit(data_rep)
    data_rep = pd.DataFrame(strat_imputer.transform(data_rep), columns=data.columns, index=data.index)
    return data_rep
    
def category_to_num(data: pd.DataFrame, out_vars: list[str]):
    return pd.DataFrame(OrdinalEncoder().fit_transform(data.drop(columns=out_vars)))

if __name__ == '__main__':
    DATA_URL = "https://github.com/ageron/data/raw/main/housing.tgz"
    DATA_DIR = "datasets"
    DATA_FILE = "housing.tgz"
    EX_DATA_DIR = "housing"
    EX_DATA_FILE = "housing.csv"
    fetch_data(DATA_URL, DATA_DIR, DATA_FILE)
    data = read_data(DATA_DIR, EX_DATA_DIR, EX_DATA_FILE)
    data['income_cat'] = np.ceil(data.median_income/1.5)
    data["income_cat"] = data["income_cat"].where(data["income_cat"]<5, 5.0)


    categorized_data = category_to_num(data, ['median_house_value'])

    print(categorized_data)
    
    # train_data, test_data = s_splitter(data, 'income_cat')
    # train_data, test_data, data = drop_col('income_cat', train_data, test_data, data)

    # print(train_data['income_cat'].value_counts()/len(train_data))
    # print(test_data['income_cat'].value_counts()/len(test_data))
    # data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, s=data["population"]/100, c="median_house_value", colormap="jet")
    # plt.show()

    # scatter_matrix_cols = ["median_house_value", "median_income", "total_rooms",
    #           "housing_median_age"]
    # scatter_matrix(data[scatter_matrix_cols])
    # plt.show()

    

