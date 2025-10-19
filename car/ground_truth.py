import numpy as np
import pickle as pkl
import os
from estimation_weighted import *

# Use this file for estimating the ground truth parameters
def main():

    res_file = 'data/ground_truth.pkl'
    if os.path.isfile(res_file):
        overwrite = input(f'Overwrite the existing ground truth parameters? [y/n]: ')
        if overwrite == 'n':
            quit()
        if overwrite not in ['y', 'n']:
            print('Please enter \'y\' or \'n\'!')    
            quit()
    print(f'Ground truth parameters will be saved in: {res_file}')

    # Load training data
    with open('data/train_real_7_1640.pkl', 'rb') as f:
        data = pkl.load(f)[0]

    y = data['y']
    X = data['X']
    y_weighted = np.array([[int(y[row] == 0), int(y[row] == 1), int(y[row] == 2)] for row in range(len(y))])
    params = fit(X, y_weighted, seed = 0)
    print(f'Ground truth parameters: {params}')

    # Save ground truth parameters
    with open(res_file, 'wb') as f:
        pkl.dump({'params': params}, f, -1)
    
if __name__ == "__main__":
    main()