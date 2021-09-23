import os

DATA_DIR = './data/CamVid/'

# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    cmd = 'git clone https://github.com/alexgkendall/SegNet-Tutorial ./data'
    os.system(cmd)
    print('Done!')
