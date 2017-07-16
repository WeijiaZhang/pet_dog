import pandas as pd
import glob
from config import *


test_id = pd.DataFrame(columns=['img_label', 'img_id'])
for file_path in glob.glob(TEST_IMG_PATH + '*.jpg'):
    file_name = file_path.split('/')
    name = file_name[-1].split('.')
    data = {'img_label': 111, 'img_id': name[0]}
    test_id = test_id.append(data, ignore_index=True)

test_id.to_csv('../../results/test_id.txt', sep='\t', header=None, index=False)

