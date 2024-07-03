#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os

def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_directory = 'output'
    output_file = os.path.join(output_directory, f'yellow_tripdata_{year:04d}-{month:02d}.parquet')

    with open('06-best-practises/model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']
    
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    df_result.to_parquet(output_file, engine='pyarrow', index=False)
    print(f'Results written to {output_file}')

if __name__ == "__main__":
    main(2023, 3)
