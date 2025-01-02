import faiss
import numpy as np
import pandas as pd
import tqdm
import os

rcdb_path = '/home/Documents/data/fvs.zip'

features = np.loadtxt(os.path.join(rcdb_path, f'features.txt'), delimiter=",", dtype=np.float32)
labels_np = np.genfromtxt(os.path.join(rcdb_path, f'label.txt'), dtype='str')
uuids_np = np.genfromtxt(os.path.join(rcdb_path, f'uuid_1.txt'), dtype='str')

df = pd.DataFrame({'upc': labels_np, 'uuid': uuids_np})
df['score'] = 0.0
df['close_to_other_product'] = 0

faiss.normalize_L2(features)
# cpu_index = faiss.IndexFlatIP(512)
# index = faiss.index_cpu_to_all_gpus(cpu_index)
index = faiss.IndexFlatIP(512)
index.add(features)

unique_upcs = df.upc.unique()
for upc in tqdm.tqdm(unique_upcs, total=len(unique_upcs), desc='Calculating distances'):
    ids = df.index[df['upc'] == upc].tolist()
    upc_count = len(ids)

    if upc_count == 1:
        continue

    _, closest_ids = index.search(features[ids], upc_count)

    ids = set(ids)
    for i, id in enumerate(ids):
        different_indices: set = set(closest_ids[i]) - ids
        score = len(different_indices) / upc_count
        df.at[id, 'score'] = score

        close_to_other_product = closest_ids[i][1] in different_indices
        df.at[id, 'upc_closest'] = labels_np[closest_ids[i][1]]
        df.at[id, 'uuids_closest'] = uuids_np[closest_ids[i][1]]
        if close_to_other_product:
            df.at[id, 'close_to_other_product'] = 1

df.to_csv('scores_result.csv', index=False)
