import numpy as np
import warnings
warnings.filterwarnings('ignore') 
from estimation_weighted import *
import copy

def flatten(X):
    X_flat =  X[1][1:] - X[0][1:]
    return X_flat
        
def flatten_full(X_list):
    res = []
    for i in range(len(X_list)):
        res.append(flatten(X_list[i]))
    return res

def fit_g(X, y, g_model = None):
    if len(np.unique(y)) == 1:
        return 0, y[0]
    # flattern X
    X_flat = flatten_full(X)

    # Train debiasing function
    if g_model is None:
        print("Using a default g_model")
        clf = MLPClassifier(solver='adam', alpha=1e-4, activation='logistic', hidden_layer_sizes=(10,5), random_state=1).fit(X_flat, y)
    else:
        g_model_sub = copy.deepcopy(g_model)
        clf = g_model_sub.fit(X_flat, y)
    return 1, clf

def train_g(X_p, y_p, z_p, g_model=None):
    # Train the de-bias functions
    g = {}
    for label in [0, 1]:
        y_p_sub = np.array([y_p[row] for row in range(len(y_p)) if z_p[row] == label])
        X_p_sub = [X_p[row] for row in range(len(y_p)) if z_p[row] == label]
        if len(y_p_sub) > 0:
            g[label] = fit_g(X_p_sub, y_p_sub, g_model = g_model)
    return g

def get_weights(X_a, z_a, g):
    pred = {}
    for j in [0,1]:
        g_type, g_func = g[j]
        if g_type == 0:
            pred[j] = np.array([[int(g_func == 0), int(g_func == 1)] for _ in range(len(X_a))])
        else:
            pred[j] = g_func.predict_proba(X_a)
    
    weights = [sum(pred[j][i] * int(z_a[i] == j) for j in [0,1])for i in range(len(z_a))]
    return weights


def aae(X_p, y_p, z_p, X_a, z_a, X_a_flat, g_model=None, concat=1, n_epochs=5000, lr=5e-4):
    
    # Human-data only estimation if there is no auxiliary data
    if len(X_a) == 0:
        w_p = np.array([[int(y_p[row] == 0), int(y_p[row] == 1)] for row in range(len(y_p))] )
        params = fit(X_p, w_p, seed=0, n_epochs=n_epochs, lr=lr)

    # Run the AAE if auxiliary data is availble
    # Step I: Train the g function using the primary dataset
    g = train_g(X_p, y_p, z_p, g_model=g_model)

    # Step II.1: Generate weights for the adjustment dataset
    w_a = get_weights(X_a_flat, z_a, g)

    # Step II.2: Estimate the parameters using the adjusted dataset
    if concat == 0:
        params = fit(X_a, w_a, seed=0, n_epochs=n_epochs, lr=lr)
    else:
        w_p = np.array([[int(y_p[row] == 0), int(y_p[row] == 1)] for row in range(len(y_p))] )
        X_c = np.concatenate([X_p, X_a], axis=0)
        w_c = np.concatenate([w_p, w_a])
        params = fit(X_c, w_c, seed=0, n_epochs=n_epochs, lr=lr)

    return params