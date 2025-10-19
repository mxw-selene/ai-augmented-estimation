import numpy as np
import os
import pickle as pkl
import warnings
warnings.filterwarnings('ignore') 
import argparse
from estimation_weighted import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

def flatten(X):
    X_flat =  X[1][1:] - X[0][1:]
    return X_flat
        
def flatten_full(X_list):
    res = []
    for i in range(len(X_list)):
        res.append(flatten(X_list[i]))
    return res

def fit_g(X, y, seed=0, g_model ='nn'):
    if len(np.unique(y)) == 1:
        return 0, y[0]
    # flattern X
    X_flat = flatten_full(X)
    # Train g function
    if g_model == 'tree':
        clf = GradientBoostingClassifier(n_estimators=5, learning_rate=0.1, max_depth=2, random_state=seed).fit(X_flat, y)
    if g_model == 'nn':
        clf = MLPClassifier(solver='adam', alpha=1e-4, activation='logistic', hidden_layer_sizes=(10, 5), random_state=1).fit(X_flat, y)
    return 1, clf

def main(dx, n_samples, n_trials, n_real, method, concat=1, g_model='nn'):

    n_real_list, n_aug_list, sample_id_list = [], [], []
    params_list = []
    n_max_aug, step_aug = 1000, 100
    n_max = n_real + n_max_aug
    rng = np.random.RandomState(0)

    # Check if the experiment results exist, overwrite if needed
    res_file = f'res/{g_model}_{method}_{n_real}_{n_max_aug}_{n_trials}.pkl'
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

    with open(f'data/ground_truth.pkl', 'rb') as f:
        true_params = pkl.load(f)['params']

    y_real, y_aug = data['y'], data['y_aug']
   
    # Generate samples paths
    participants_samples, row_samples, g_samples, w_a_samples = [], [], [], []
    for i in range(n_trials):
        participants = rng.choice(int(n_max/5), size = int((n_real + n_max_aug)/5), replace=False)
        rows = []
        for j in participants:
            rows += list(range(j*5, (j*5)+5))
        participants_samples.append(participants)
        row_samples.append(rows)

        real_rows = rows[0:n_real]
        y_p = np.array([y_real[row] for row in real_rows])
        w_p = np.array([[int(y_real[row] == 0), int(y_real[row] == 1)] for row in real_rows])
        z_p = np.array([y_aug[row] for row in real_rows])
        X_p = [data['X'][row] for row in real_rows]

        # Train the g functions
        g_function = {}
        for label in [-1, 0, 1]:
            y_p_sub = np.array([y_p[row] for row in range(len(y_p)) if z_p[row] == label])
            X_p_sub = [X_p[row] for row in range(len(y_p)) if z_p[row] == label]
            if len(y_p_sub) > 0:
                g_function[label] = fit_g(X_p_sub, y_p_sub, seed = 0, g_model = g_model)
        g_samples.append(g_function)

        # Construct auxiliary datasets
        aug_rows = rows[n_real:n_real + n_max_aug]
        z_a = np.array([y_aug[row] for row in aug_rows])
        X_a = [data['X'][row] for row in aug_rows]

        # Compute adjustment weights for the auxiliary dataset
        w_a = []
        for row in range(len(z_a)):
            if z_a[row] in g_function:
                g_type, g_func = g_function[z_a[row]]
                if g_type == 0:
                    weights = np.array([int(g_func == 0), int(g_func == 1)])
                else:
                    X_flat = flatten(X_a[row])
                    classes = g_func.classes_
                    weights = g_func.predict_proba([X_flat])[0]
                    for j, cl in enumerate([0, 1]):
                        if cl not in classes:
                            weights = np.insert(weights, j, 0)
                w_a.append(weights)
            else:
                w_a.append(np.array([int(z_a[row] == 0), int(z_a[row] == 1)]))
        w_a_samples.append(w_a)

    print('m\tn\tMAPE(%)\n--------------------------------')
    # Load the sampled datasets and calculate the MAPE for each (n_real, n_aug)
    for n_aug in np.arange(0, n_max_aug + step_aug, step_aug):
        # Skip if no real data and no augmented data
        if n_real == 0 and n_aug == 0:
            continue       

        # Run AAE and calculate average MAPE for each (n_real, n_aug)
        mape_sum = 0
        for i in range(n_trials):     
            n_real_list.append(n_real)
            n_aug_list.append(n_aug)
            sample_id_list.append(i)
            
            real_rows = row_samples[i][0:n_real]
            aug_rows = row_samples[i][n_real:n_real+n_aug]
        
            y_p = np.array([y_real[row] for row in real_rows])
            w_p = np.array([[int(y_real[row] == 0), int(y_real[row] == 1)] for row in real_rows] )                   
            X_p = [data['X'][row] for row in real_rows]
        
            # Generate primary datasets
            w_a = np.array(w_a_samples[i][0:n_aug])
            X_a = [data['X'][row] for row in aug_rows]

            # Train with adjusted data
            if n_aug != 0 and concat == 0:
                params = fit(X_a, w_a, seed = 0)
            elif n_aug == 0:
                params = fit(X_p, w_p, seed = 0)
            else:
                X_c = X_a + X_p
                w_c = np.concatenate([w_a, w_p])
                params = fit(X_c, w_c,  seed = 0)

            # Calculate MAPE
            mape = np.mean(np.abs((params - true_params)/(true_params + 1)) * 100)
            mape_sum += mape
            params_list.append(params)
        print(f'{n_real}\t{n_aug}\t{mape_sum/n_trials:.2f}')

    # Save experiment results
    with open(res_file, 'wb') as f:
        pkl.dump({'params_list': params_list, 'n_real_list': n_real_list, 'n_aug_list': n_aug_list, 'sample_id_list': sample_id_list}, f, -1)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx', '-dx', type=int, default=11)
    parser.add_argument('--n_samples', '-ns', type=int, default=1200)
    parser.add_argument('--n_trials', '-nt', type=int)
    parser.add_argument('--n_real', '-nr', type=int)
    parser.add_argument('--g_model', '-gm', type=str, choices = ['tree', 'nn'], default='nn')
    parser.add_argument('--method', '-method', type=str)
    parser.add_argument('--concat', '-cc', type=int, default=1)
    args = parser.parse_args()
    main(args.dx, args.n_samples, args.n_trials, args.n_real, args.method, args.concat, args.g_model)