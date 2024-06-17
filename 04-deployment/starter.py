import pickle
import pandas as pd
import numpy as np
import os
import sys


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)




categorical = ['PULocationID', 'DOLocationID']
year = 2023
month = 4

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')  
    return df



df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-04.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


y_pred

print(np.mean(y_pred))


std_dev = np.std(y_pred)
print(f'What is the standard deviation of the predicted duration for this dataset? {std_dev:.2f}')



df['ride_id'] = f'{2023:04d}/{2:02d}_' + df.index.astype('str')




df




df_result = pd.DataFrame()
df_result['duration'] = df['duration']
df_result['ride_id'] = df['ride_id']
df_result



df_result.to_parquet(
    'output.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)




len(df_result)




file_path = './output.parquet'
file_size_bytes = os.path.getsize(file_path)

file_size_kb = file_size_bytes / 1024
file_size_mb = file_size_bytes / (1024 ** 2)
file_size_gb = file_size_bytes / (1024 ** 3)


print(f'Size of the output.parquet file is: {file_size_mb:.2f} MB')






