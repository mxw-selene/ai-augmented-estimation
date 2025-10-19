import numpy as np
import os 
import pickle as pkl
import warnings
warnings.filterwarnings('ignore') 
import argparse
from estimation_weighted import *


def main(dx, n_samples, n_trials, n_real, method, n_max_aug=500, step_aug=100, concat=1):
    n_real_list, n_aug_list, sample_id_list = [], [], []
    params_list = []
    n_max = n_real + n_max_aug
    rng = np.random.RandomState(0)

    # Check if the experiment results exist, overwrite if needed
    res_file = f'res/naive_{method}_{n_real}_{n_max_aug}_{n_trials}.pkl'
    if os.path.isfile(res_file):
        overwrite = input(f'Overwrite the exp results in {res_file}? [y/n]: ')
        if overwrite == 'n':
            quit()
        if overwrite not in ['y', 'n']:
            print('Please enter \'y\' or \'n\'!')
            quit()
    print(f'Experiment results will be saved in: {res_file}')

    # Load training data
    with open(f'data/train_{method}_{dx}_{n_samples}.pkl', 'rb') as f:
        data = pkl.load(f)[0]
    y_real, y_aug = data['y'], data['y_aug']

    # Load ground truth parameters
    with open(f'data/ground_truth.pkl', 'rb') as f:
        true_params = pkl.load(f)['params']
    param_indices = [0,1,3,4,5,6] # Remove the third column due to small its magnitude, which makes the relative bias reduction not meaningful

    # Generate samples paths
    participants_samples, row_samples = [], []
    for i in range(n_trials):
        participants = rng.choice(int(n_max/10), size = int((n_real + n_max_aug)/10), replace=False)
        rows = []
        for j in participants:
            rows += list(range(j*10, (j*10)+10))
        participants_samples.append(participants)
        row_samples.append(rows)

    print('m\tn\tMAPE(%)\n--------------------------------')
    # Load the sampled datasets and calculate the MAPE for each (n_real, n_aug)
    for n_aug in np.arange(0, n_max_aug + step_aug, step_aug):
        if n_real == 0 and n_aug == 0:
            continue       
        mape_sum = 0        
        for i in range(n_trials):     
            n_real_list.append(n_real)
            n_aug_list.append(n_aug)
            sample_id_list.append(i)
            
            real_rows = row_samples[i][0:n_real]
            aug_rows = row_samples[i][n_real:n_real+n_aug]

            # Generate validation dataset        
            w_p = np.array([[int(y_real[row] == 0), int(y_real[row] == 1), int(y_real[row] == 2)] for row in real_rows] )                   
            X_p = [data['X'][row] for row in real_rows]
        
            # Generate primary dataset
            w_a = np.array([[int(y_aug[row] == 0), int(y_aug[row] == 1), int(y_aug[row] == 2)] for row in aug_rows])
            X_a = [data['X'][row]  for row in aug_rows]

            # Train with adjusted data
            if n_aug != 0 and concat == 0:
                params = fit(X_a, w_a, seed = 0)
            elif n_aug == 0:
                params = fit(X_p, w_p, seed = 0)
            else:
                X_c = X_a + X_p
                if len(w_p) > 0:
                    w_c = np.concatenate([w_a, w_p])
                else:
                    w_c = w_a
                params = fit(X_c, w_c, seed = 0)

            mape = np.mean(np.abs((params[param_indices] - true_params[param_indices])/(true_params[param_indices])) * 100)
            mape_sum += mape
            params_list.append(params)
        print(f'{n_real}\t{n_aug}\t{mape_sum/n_trials:.2f}')

    # Save experiment results
    with open(res_file, 'wb') as f:
        pkl.dump({'params_list': params_list, 'n_real_list': n_real_list, 'n_aug_list': n_aug_list, 'sample_id_list': sample_id_list}, f, -1)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx', '-dx', type=int, default=7)
    parser.add_argument('--n_samples', '-ns', type=int, default=1200)
    parser.add_argument('--n_trials', '-nt', type=int)
    parser.add_argument('--n_real', '-nr', type=int)
    parser.add_argument('--method', '-method', type=str)
    parser.add_argument('--concat', '-cc', type=int, default=1)
    parser.add_argument('--n_max_aug', '-nma', type=int, default=500)
    parser.add_argument('--step_aug', '-sa', type=int, default=100)
    args = parser.parse_args()
    main(args.dx, args.n_samples, args.n_trials, args.n_real, args.method, args.n_max_aug, args.step_aug, args.concat)