import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *  
from shutil import copyfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# List of datasets
datasets = ['SMD', 'SWaT', 'SMAP', 'MSL', 'NAB']

# Function to load and save data for the given category, filename, dataset, and dataset folder
def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

# Function to load and save labeled data
def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
        temp[start-1:end-1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

# Function to normalize the input array 'a' using min-max normalization
def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)

# Function to normalize the input array 'a' with custom min and max values
def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a

# Function to normalize the input array 'a' with handling for missing or NaN values
def normalize3(a, min_a=None, max_a=None):
    if np.isnan(a).any():
        imputer = SimpleImputer(strategy='mean')
        a = imputer.fit_transform(a)

    if min_a is not None and max_a is not None:
        normalized_a = (a - min_a) / (max_a - min_a + 0.0001)
        return normalized_a, min_a, max_a

    scaler = StandardScaler()
    normalized_a = scaler.fit_transform(a)
    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return normalized_a, min_a, max_a

# Instantiate a scaler object
scaler = StandardScaler()

# Function to convert a Pandas DataFrame to numpy array after selecting specific columns
def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)

# Function to load data for a specific dataset
def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'SMD':
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
    elif dataset == 'NAB':
        dataset_folder = 'data/NAB'
        file_list = os.listdir(dataset_folder)
        with open(os.path.join(dataset_folder, 'labels.json')) as f:
            labeldict = json.load(f)
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            df = pd.read_csv(os.path.join(dataset_folder, filename))
            vals = df.values[:, 1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in labeldict['realKnownCause/' + filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                labels[index-4:index+4] = 1
            min_temp, max_temp = np.min(vals), np.max(vals)
            vals = (vals - min_temp) / (max_temp - min_temp)
            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            fn = filename.replace('.csv', '')
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'
        file = os.path.join(dataset_folder, 'series.json')
        df_train = pd.read_json(file, lines=True)[['val']][3000:6000]
        df_test = pd.read_json(file, lines=True)[['val']][7000:12000]
        train, min_a, max_a = normalize2(df_train.values)
        test, _, _ = normalize2(df_test.values, min_a, max_a)
        labels = pd.read_json(file, lines=True)[['noti']][7000:12000] + 0
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = 'data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test, min_a, max_a)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i+1], :] = 1
            np.save(f'{folder}/{fn}_labels.npy', labels)
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')

# Main block to execute when the script is run
if __name__ == '__main__':
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is a space-separated list of {datasets}")
