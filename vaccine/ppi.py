import numpy as np
import os
import pickle as pkl
import warnings
warnings.filterwarnings('ignore') 
import argparse
from estimation_weighted import *
from ppi_py import ppi_logistic_pointestimate

optimizer_options = {
                    "ftol": 1e-5,
                    "gtol": 1e-5,
                    "maxls": 10000,
                    "maxiter": 10000,
                }

def flatten(X):
    X_flat =  X[1][1:] - X[0][1:]
    return X_flat
        
def flatten_full(X_list):
    res = []
    for i in range(len(X_list)):
        res.append(flatten(X_list[i]))
    return np.array(res)

def main(dx, n_samples, n_trials, n_real, method, db_model='ppi'):

    n_real_list, n_aug_list, sample_id_list = [], [], []
    params_list = []
    n_max_aug, step_aug = 1000, 100
    n_max = n_real + n_max_aug
    rng = np.random.RandomState(0)

    # Check if the experiment results exist, overwrite if needed
    res_file = f'res/{db_model}_{method}_{n_real}_{n_max_aug}_{n_trials}.pkl'
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

    X, y_real, y_aug = data['X'], data['y'], data['y_aug']
    with open(f'data/ground_truth.pkl', 'rb') as f:
        true_params = pkl.load(f)['params']
   
    # Generate samples paths
    participants_samples, row_samples = [], []
    for i in range(n_trials):
        participants = rng.choice(int(n_max/5), size = int((n_real + n_max_aug)/5), replace=False)
        rows = []
        for j in participants:
            rows += list(range(j*5, (j*5)+5))
        participants_samples.append(participants)
        row_samples.append(rows)

    # Load the sampled datasets and calculate the MAPE for each (n_real, n_aug)
    print('m\tn\tMAPE(%)\n--------------------------------')
    for n_aug in np.arange(0, n_max_aug + step_aug, step_aug):
        if n_real == 0 and n_aug == 0:
            continue       

        # Run PPI and calculate average MAPE for each (n_real, n_aug)
        mape_sum = 0 
        for i in range(n_trials):     
            params =  10000 * np.ones(X[0].shape[1] - 1) # Initialize with a large value to signal singularity issues in the estimation
            try:
                n_real_list.append(n_real)
                n_aug_list.append(n_aug)
                sample_id_list.append(i)
                
                real_rows = row_samples[i][0:n_real]
                aug_rows = row_samples[i][n_real:n_real+n_aug]
            
                y_p = np.array([y_real[row] for row in real_rows])
                z_p = np.array([y_aug[row] for row in real_rows])
                X_p = [X[row] for row in real_rows]
                if n_aug == 0:
                    w_p = np.array([[int(y_real[row] == 0), int(y_real[row] == 1)] for row in real_rows] )
                    params = fit(X_p, w_p, seed = 0)
                else:
                    X_p = flatten_full(X_p)
                    # Sample auxiliary datasets
                    z_a = np.array([y_aug[row] for row in aug_rows])
                    X_a = np.array([X[row] for row in aug_rows])
                    X_a = flatten_full(X_a)
                    if db_model == 'ppi':
                        params = ppi_logistic_pointestimate(X_p, y_p, z_p, X_a, z_a, \
                            lam=1, optimizer_options=optimizer_options)
                    elif db_model == 'ppi++':
                        params = ppi_logistic_pointestimate(X_p, y_p, z_p, X_a, z_a, \
                            optimizer_options=optimizer_options)
            except Exception as e:
                pass

            mape = np.mean(np.abs((params - true_params)/(true_params + 1)) * 100)
            mape_sum += mape
            params_list.append(params)
        print(f'{n_real}\t{n_aug}\t{"{:.2f}".format(mape_sum/n_trials) if mape_sum/n_trials < 1000 else "-"}')

    # Save experiment results
    with open(res_file, 'wb') as f:
        pkl.dump({'params_list': params_list, 'n_real_list': n_real_list, 'n_aug_list': n_aug_list, 'sample_id_list': sample_id_list}, f, -1)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx', '-dx', type=int, default=11)
    parser.add_argument('--n_samples', '-ns', type=int, default=1200)
    parser.add_argument('--n_trials', '-nt', type=int)
    parser.add_argument('--n_real', '-nr', type=int)
    parser.add_argument('--db_model', '-db', type=str, choices = ['ppi', 'ppi++'], default='ppi')
    parser.add_argument('--method', '-method', type=str)
    args = parser.parse_args()
    main(args.dx, args.n_samples, args.n_trials, args.n_real, args.method, args.db_model)