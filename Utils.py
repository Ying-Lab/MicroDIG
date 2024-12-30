import numpy as np
import scipy
import torch
import random
import scipy.sparse as sp

def adjust_concentrations(np_data):
    """Change the scale of observed concentrations.
    """
    con = np_data.sum(axis=0)
    C = 1 / np.mean(con)

    np_data_adjusted = np_data*C
    if (((np_data_adjusted.max(axis=1)<0.01).astype(int).sum())/np_data_adjusted.shape[0])>0.1:
        np_data_adjusted = np_data_adjusted*5

    return np_data_adjusted


def least_squares_lotka_volterra(np_data,t_step):
    ntaxa = np_data.shape[0]
    t_num = np_data.shape[1]
    predictors = [[] for n in range(ntaxa)]
    outcomes = [[] for n in range(ntaxa)]
    for t in range(1,t_num):
        delT = t_step[t] - t_step[t-1]
        xt  = np_data[:,t]
        xt0 = np_data[:,t-1]

        xt_xt_T = np.outer(xt, xt)
        for n in range(ntaxa):
            outcomes[n].append((xt - xt0)[n] / delT)
            tmp = np.concatenate( (xt_xt_T[n], [xt[n]]))
            predictors[n].append(tmp)
    predictors = np.array(predictors)
    outcomes = np.array(outcomes)
    Wij = np.zeros((ntaxa,ntaxa))
    Ui = np.zeros(ntaxa)
    for n in range(ntaxa):
        P = predictors[n]
        Z = np.expand_dims(outcomes[n], axis=1)
        parameters = np.linalg.pinv(P.T.dot(P) + 0.001*np.eye(P.shape[1])).dot(P.T).dot(Z)
        Wij[:,n] = parameters[:ntaxa].flatten()
        Ui[n] = parameters[ntaxa]
    return Wij,Ui


def compute_simulation_parameters_uniform(np_data,t_index):
    Wij,Ui = least_squares_lotka_volterra(np_data,t_index)
    ntaxa = Wij.shape[0]

    Ui = np.abs(Ui)
    Wij[np.diag_indices(ntaxa)] = -np.abs(Wij[np.diag_indices(ntaxa)])

    Wij_self_min = np.percentile(Wij[np.diag_indices(ntaxa)].flatten(),0)
    Wij_self_max = np.percentile(Wij[np.diag_indices(ntaxa)].flatten(),100)
    
    Wij_interact_min = np.percentile(Wij[~np.eye(ntaxa,dtype=bool)].flatten(),0)
    Wij_interact_max = np.percentile(Wij[~np.eye(ntaxa,dtype=bool)].flatten(),100)

    Ui_min = np.percentile(Ui,0)
    Ui_max = np.percentile(Ui,100)

    return Wij_self_min,Wij_self_max,Wij_interact_min,Wij_interact_max,Ui_min,Ui_max


#------------------------------------------------------------------------------------------------------------------------------

def compute_simulation_parameters_normal(np_data,t_index):
    Wij,Ui = least_squares_lotka_volterra(np_data,t_index)
    ntaxa = Wij.shape[0]

    Ui = np.abs(Ui)
    Wij[np.diag_indices(ntaxa)] = -np.abs(Wij[np.diag_indices(ntaxa)])

    Wij_self_mean = np.mean(Wij[np.diag_indices(ntaxa)].flatten())

    Wij_self_var = np.var(Wij[np.diag_indices(ntaxa)].flatten())
    Wij_interact_var = np.var(Wij[~np.eye(ntaxa,dtype=bool)].flatten())

    Ui_mean = np.mean(Ui)
    Ui_var = np.var(Ui)

    initial_cond = np_data[:,0]
    initial_mean = np.mean(initial_cond)
    initial_var = np.var(initial_cond)
    return Wij_self_mean,Wij_self_var,Wij_interact_var,Ui_mean,Ui_var,initial_mean,initial_var

def fix_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)