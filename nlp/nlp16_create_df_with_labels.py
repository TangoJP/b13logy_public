import pandas as pd
import numpy as np
import pickle

# Read in the DataFrame
df = pd.read_csv('base_data/pride_table.csv',
                usecols=[
                    'dataset_id', 
                    'sample_protocol', 
                    'data_protocol',
                    'description',
                    'quant_methods'])

df = df.dropna().reset_index(drop=True)

# Replace 'Not available' texts with NaN
df.loc[:, 'sample_protocol'] = df.loc[:, 'sample_protocol'].replace({'Not available': np.NaN, 'nan':np.NaN})
df.loc[:, 'data_protocol'] = df.loc[:, 'data_protocol'].replace({'Not available': np.NaN, 'nan':np.NaN})
df.loc[:, 'description'] = df.loc[:, 'description'].replace({'Not available': np.NaN, 'nan':np.NaN})

# Drop rows that have null text fields
df.dropna(subset=['sample_protocol', 'data_protocol', 'description'], inplace=True)
print('# rows after dropna:', len(df))
    
df['silac'] = df.quant_methods.str.contains('silac').astype(int)
df['ms1_label_free'] = df.quant_methods.str.contains('ms1 intensity based label-free quantification method').astype(int)
df['spectrum_counting'] = df.quant_methods.str.contains('spectrum counting').astype(int)
df['tmt'] = df.quant_methods.str.contains('tmt').astype(int)
df['itraq'] = df.quant_methods.str.contains('itraq').astype(int)
df['label_free'] = df.quant_methods.str.contains('label free').astype(int)

with open('base_data/pride_quant_labeled.pickle', 'wb') as outfile:
    pickle.dump(df, outfile)
print('DF serialized and saved.')