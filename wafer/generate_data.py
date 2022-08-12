import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import h5py

# Please download the original data from https://www.kaggle.com/code/ashishpatel26/wm-811k-wafermap/data.
pickle_path = '/path/to/LSWMD.pkl'
df = pd.read_pickle(pickle_path)
df = df.drop(['waferIndex'], axis=1)
df['failureNum'] = df.failureType.apply(lambda y: 9 if len(y) == 0 else y)
df['trainTestNum'] = df.trianTestLabel

mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3, 'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8}
mapping_traintest = {'Training': 0, 'Test': 1}
df = df.replace({'failureNum': mapping_type, 'trainTestNum': mapping_traintest})

print("Starting to save data...")
df_in = df.reset_index()

dim = 32
dimx, dimy = dim, dim
path = '../data/WM-811k' + '/' + str(dim)
if not os.path.exists(path):
    os.makedirs(path)
targets = df_in['failureNum'].to_numpy()
data = []
for array in tqdm(df_in['waferMap']):
    image = Image.fromarray((array * 127.5).astype('uint8'), 'L')
    image = image.resize((dimx, dimy))
    data.append(np.array(image))
data = np.array(data)

train_idx = np.where(np.array(((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8) & (df_in['trainTestNum'] == 0)).array))[0]
test_idx = np.where(np.array(((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8) & (df_in['trainTestNum'] == 1)).array))[0]
label_idx = np.where(np.array((((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8))).array))[0]
unlabel_idx = np.where(np.array((~((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8))).array))[0]
train_unlabel_idx = np.where(((~((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8))) | ((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8) & (df_in['trainTestNum'] == 0))).array)[0]
test_unlabel_idx = np.where(((~((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8))) | ((df_in['failureNum'] >= 0) & (df_in['failureNum'] <= 8) & (df_in['trainTestNum'] == 1))).array)[0]

# train
database = h5py.File(path+'/train.h5', 'w')
temptar = targets[train_idx]
tempdat = data[train_idx]
database.create_dataset('data', data=tempdat, compression='gzip')
database.create_dataset('targets', data=temptar, compression='gzip')
database.close()

# test
database = h5py.File(path+'/test.h5', 'w')
temptar = targets[test_idx]
tempdat = data[test_idx]
database.create_dataset('data', data=tempdat, compression='gzip')
database.create_dataset('targets', data=temptar, compression='gzip')
database.close()

# label
database = h5py.File(path+'/labeled.h5', 'w')
temptar = targets[label_idx]
tempdat = data[label_idx]
database.create_dataset('data', data=tempdat, compression='gzip')
database.create_dataset('targets', data=temptar, compression='gzip')
database.close()

# # unlabel
# database = h5py.File(path+'/unlabel.h5', 'w')
# tempdat = data[unlabel_idx]
# database.create_dataset('data', data=tempdat, compression='gzip')
# database.close()

# # train+unlabel
# database = h5py.File(path+'/train+unlabel.h5', 'w')
# tempdat = data[train_unlabel_idx]
# database.create_dataset('data', data=tempdat, compression='gzip')
# database.close()

# # test+unlabel
# database = h5py.File(path+'/test+unlabel.h5', 'w')
# tempdat = data[test_unlabel_idx]
# database.create_dataset('data', data=tempdat, compression='gzip')
# database.close()
