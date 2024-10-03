import numpy as np
import pandas as pd
import tarfile
from pathlib import Path
import zipfile
import urllib.request as req
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder 


def fetch_data(dir_url: str|None, dir_path: str, file_name: str, extracted: bool = True):
    file_path = Path(dir_path) / file_name

    if file_path.is_file() or extracted:
        print("File path exists")
        return

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    if dir_url is not None:
        req.urlretrieve(dir_url, file_path)

    if not zipfile.is_zipfile(file_path):
        with tarfile.open(file_path) as tarball:
            tarball.extractall(path=dir_path)
            return

    with zipfile.ZipFile(file_path) as zip:
        zip.extractall(path=dir_path)

def read_data(dir_path: str, filename: str):
    path = Path(dir_path) / filename
    data = pd.read_csv(filepath_or_buffer=path)
    return data

def vectorizer(data: pd.DataFrame, col: str):
    docs: list[str] = data[col].to_list()
    model = Doc2Vec(vector_size=150, window=2, epochs=50, min_count=1)

    data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(docs) if isinstance(doc, str)]

    model.build_vocab(data)
    if len(model.wv.index_to_key) == 0:
        raise RuntimeError("Vocabulary not built properly.")
    model.train(data, total_examples=model.corpus_count, epochs=model.epochs)

    vectors = [model.infer_vector(doc.split()) for doc in docs if isinstance(doc, str)]
    return pd.DataFrame(vectors)

def get_nullorna_count_cls(data: pd.DataFrame):
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

def fill_na_strategy(data: pd.DataFrame, fill_strat="remove", remove_cols: list[str]=[]):
    data_rep = data
    if len(remove_cols) > 0:
        data_rep = data.drop(columns=remove_cols)

    if fill_strat == "remove":
        data_rep = data.dropna()
        return data_rep

    data_rep = data_rep.select_dtypes(include=[np.number])
    print(data_rep.head())
    strat_imputer = SimpleImputer(strategy=fill_strat)
    strat_imputer.fit(data_rep)
    data_rep = pd.DataFrame(strat_imputer.transform(data_rep), columns=data.columns, index=data.index)
    return data_rep
    
def category_to_num(data: pd.DataFrame, out_vars: list[str]):
    return pd.DataFrame(OrdinalEncoder().fit_transform(data.drop(columns=out_vars)))


if __name__ == '__main__':
    fetch_data(None, 'datasets', 'archive.zip')
    train_data = read_data('datasets', 'twitter_training.csv')
    test_data = read_data('datasets', 'twitter_validation.csv')

    print(train_data.shape)
    print(test_data.shape)

    train_null_cols = get_nullorna_count_cls(train_data).index
    test_null_cols = get_nullorna_count_cls(test_data).index
    print(test_null_cols)

    train_data = fill_na_strategy(train_data, remove_cols=train_data.columns.drop(train_null_cols))    
    test_data = fill_na_strategy(test_data, remove_cols=test_data.columns.drop(test_null_cols))    

    print(train_data.shape)
    print(test_data.shape)

    train_data['tweet'] = vectorizer(train_data, 'tweet')
    test_data['tweet'] = vectorizer(test_data, 'tweet')

    print(train_data.head())
    print(test_data.head())

    hist = train_data.sentiment.value_counts()
    encodable_cols = hist.index.to_list()

    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_data_y = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['sentiment']]), columns=encodable_cols)
    test_data_y = pd.DataFrame(one_hot_encoder.fit_transform(test_data[['sentiment']]), columns=encodable_cols)

    train_data_x = train_data.drop(columns=['sentiment'])
    test_data_x = test_data.drop(columns=['sentiment'])
    
    print(train_data_x.head())
    rint(test_data_y.head())
