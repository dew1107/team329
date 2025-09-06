import pandas as pd
import ast
import random
import os

# Load the datasets
train_captions_df = pd.read_csv('../../train_captions.csv') # data type 에 따라 train/valid/test 로 변경
train_concepts_df = pd.read_csv('../../train_concepts.csv')
train_concepts_manual_df = pd.read_csv('../../train_concepts_manual.csv')
cui_mapping_diseases_df = pd.read_csv('../../cui_mapping_diseases.csv')

# Step 1: Combine the two concepts dataframes
combined_concepts_df = pd.merge(train_concepts_df, train_concepts_manual_df, on='ID', how='outer', suffixes=('_auto', '_manual'))

# Fill NaN values with empty strings
combined_concepts_df['CUIs_auto'] = combined_concepts_df['CUIs_auto'].fillna('')
combined_concepts_df['CUIs_manual'] = combined_concepts_df['CUIs_manual'].fillna('')

# Combine the CUIs from both columns
def combine_cuis(row):
    cuis_list = []
    # Check if the column value is a string before splitting
    if isinstance(row['CUIs_auto'], str) and row['CUIs_auto']:
        cuis_list.extend(row['CUIs_auto'].split(';'))
    if isinstance(row['CUIs_manual'], str) and row['CUIs_manual']:
        cuis_list.extend(row['CUIs_manual'].split(';'))
    # Remove duplicates and join back
    return ';'.join(sorted(list(set(cuis_list))))

combined_concepts_df['all_cuis'] = combined_concepts_df.apply(combine_cuis, axis=1)

# Step 2: Create a CUI to Canonical name mapping dictionary
cui_mapping_dict = dict(zip(cui_mapping_diseases_df['CUI'], cui_mapping_diseases_df['Canonical name']))

# Step 3: Merge with the captions dataframe
master_df = pd.merge(train_captions_df, combined_concepts_df[['ID', 'all_cuis']], on='ID', how='inner')

# Step 4: Process CUIs and map to labels
def get_disease_labels(cui_string):
    if not isinstance(cui_string, str) or not cui_string:
        return []
    cuis = cui_string.split(';')
    labels = []
    for cui in cuis:
        if cui in cui_mapping_dict:
            labels.append(cui_mapping_dict[cui])
    return sorted(list(set(labels)))

master_df['Disease_Labels'] = master_df['all_cuis'].apply(get_disease_labels)

# Step 5: Filter out rows with no labels and format the labels
master_df = master_df[master_df['Disease_Labels'].apply(len) > 0].copy()
master_df['Disease_Labels'] = master_df['Disease_Labels'].apply(lambda x: str(x))
master_df.drop(columns=['all_cuis'], inplace=True)

# Check the final dataframe before distribution
print("\nFinal master_df info:")
master_df.info()
print("\nFinal master_df head:")
print(master_df.head())

# Step 6: Data distribution
random.seed(42)
num_total_ids = len(master_df['ID'].unique())
sample_size = min(5000, num_total_ids)
print(f"Total unique IDs after filtering: {num_total_ids}")

all_ids = master_df['ID'].unique()
clients_data = {}
for i in range(30):
    current_sample_size = min(sample_size, num_total_ids)
    clients_data[i] = random.sample(list(all_ids), current_sample_size)

# Function to save data for a client, creating a directory for each one
def save_client_data(client_id, data_type, df):
    dir_path = f'../../_prepared/client_{client_id}'
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f'client_{client_id}_train_{data_type}.csv')
    df.to_csv(file_path, index=False)
    print(f"Saved {file_path} with {len(df)} rows.")

# Distribute and save
for i in range(30):
    client_ids = clients_data[i]
    client_df = master_df[master_df['ID'].isin(client_ids)]

    if i <= 25:
        # Clients 0-25 get both image and text data
        save_client_data(i, 'image', client_df)
        save_client_data(i, 'text', client_df)
    elif i <= 27:
        # Clients 26-27 get only image data
        save_client_data(i, 'image', client_df)
    else: # i <= 29
        # Clients 28-29 get only text data
        save_client_data(i, 'text', client_df)