from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.base import clone

from copy import deepcopy

from collections import OrderedDict

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
jax.default_device(jax.devices('cpu')[0])

import jax.numpy as jnp
import numpy as np
import pandas as pd

import itertools

import matplotlib.pyplot as plt

from weatherstations_clusterer.dag_kl import dag_kl
from weatherstations_clusterer.better_graphs import process_graph

# the above stuff is likely boilerplate but I have no idea what precisely it does

def visualize(data, y, y_hat):
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # Double the width to accommodate two plots
    ax[0].scatter(data_reduced[:, 0], data_reduced[:, 1], c=y_hat, edgecolor='k', s=50, cmap='rainbow')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')
    ax[0].set_title('PCA, Predicted Labels by GMM')
    ax[0].grid(True)
    scatter = ax[1].scatter(data_reduced[:, 0], data_reduced[:, 1], c=y, edgecolor='k', s=50, cmap='viridis')
    ax[1].set_xlabel('Principal Component 1')
    ax[1].set_ylabel('Principal Component 2')
    ax[1].set_title('PCA Ground Truth')
    ax[1].grid(True)
    plt.show()


# I am like 93% sure that this is the part where they check the redundancy condition
def check_epsilons(gmm, n_samples, axes_to_keep):
    data, labels = gmm.sample(n_samples)
    data_castrated = data[:, axes_to_keep]

    gmm_castrated = GaussianMixture(n_components=len(axes_to_keep), covariance_type='diag')

    gmm_castrated.weights_ = gmm.weights_
    gmm_castrated.covariances_ = gmm.covariances_[:, axes_to_keep]
    gmm_castrated.means_ = gmm.means_[:, axes_to_keep]
    gmm_castrated.precisions_cholesky_ = jnp.sqrt(1 / gmm_castrated.covariances_)
    gmm_castrated.converged = True

    p_L_X = gmm.predict_proba(data)
    p_L_Xs = gmm_castrated.predict_proba(data_castrated)

    p_X = np.exp(gmm.score_samples(data))

    # edkl = p_X @ np.einsum("xl,xl->x", p_L_X, (np.log(p_L_X) - np.log(p_L_Xs)))
    # the above line was commented out when it got to me
    
    edkl = np.einsum("xl,xl->x", p_L_X, (np.log(p_L_X) - np.log(p_L_Xs))).mean()

    edkl = edkl / jnp.log(2)

    if edkl > 50:
        print("woe unto all")

    print("E_x[Dkl(P[(L|X) || (L|Xs)])] = ", edkl)

    return edkl
    
full_axes = list(range(12))  # makes the list of the 12 ways a single axis might be dropped
# print(full_axes)
dropped_axis_list = []

for i in range(12):
    tempfull = full_axes.copy()
    tempfull.remove(i)
    dropped_axis_list.append(tempfull)
    # print(tempfull)

# print(dropped_axis_list)

def main():

    n_gmm_components = 3
    # original was 3 (species of flower); NCEI divides weather stations ~geographically into 6, ultimately I ended up using Kepler Objects instead which... class into Candidates, Confirmeds, and False Positives. F.
    covariance_type = 'diag'
    init_params = 'random' # default: 'kmeans'

    data_df = pd.read_csv("cumulative_2024.06.26_12.15.36")  # added this dataset in
    data = data_df[[c for c in data_df.columns[:-1]]].to_numpy(dtype=np.float32)
    y = data_df['category'].map({"confirmed": 0, "candidate": 1, "false positive": 2}).values  # this line is almost the right shape but wrong as is

    gmm = GaussianMixture(n_components=n_gmm_components, random_state=0,
                          covariance_type=covariance_type, init_params=init_params).fit(data)
    y_hat = gmm.predict(data)

    log_probs = jnp.log(gmm.predict_proba(data))
    probs = gmm.predict_proba(data)
    probs[jnp.isinf(log_probs)] = 0.0
    log_probs = log_probs.at[jnp.isinf(log_probs)].set(0.0)
    entropy_l_given_x = -(probs*log_probs).sum(axis=1).mean() / jnp.log(2)
    print("Entropy of P[L|X]: ", entropy_l_given_x)

    for axes_to_keep in generate_combinations(list(range(data.shape[1]))):
        print(axes_to_keep)
        check_epsilons(gmm, n_samples=len(data), axes_to_keep=axes_to_keep)

    redundancy_error = 0
    for axes_to_keep in dropped_axis_list:
        redundancy_error += check_epsilons(gmm, n_samples=len(data), axes_to_keep=axes_to_keep)
    print("\nSum of redundancy errors for weak invar: ", redundancy_error)

    print("\nIsomorphism bound: ", redundancy_error + entropy_l_given_x*2)

    # visualize(data, y, y_hat)


    print('\n\n=================\n')

    gmm2 = GaussianMixture(n_components=n_gmm_components, random_state=1,
                          covariance_type=covariance_type, init_params=init_params).fit(data)
    y_hat_2 = gmm2.predict(data)

    log_probs_2 = jnp.log(gmm2.predict_proba(data))
    probs_2 = gmm2.predict_proba(data)
    probs_2[jnp.isinf(log_probs_2)] = 0.0
    log_probs_2 = log_probs_2.at[jnp.isinf(log_probs_2)].set(0.0)
    entropy_l_given_x_2 = -(probs_2 * log_probs_2).sum(axis=1).mean() / jnp.log(2)
    print("Entropy of P[L|X]: ", entropy_l_given_x_2)

    for axes_to_keep in generate_combinations(list(range(data.shape[1]))):
        print(axes_to_keep)
        check_epsilons(gmm2, n_samples=len(data), axes_to_keep=axes_to_keep)

    redundancy_error = 0
    for axes_to_keep in dropped_axis_list:
        redundancy_error += check_epsilons(gmm2, n_samples=len(data), axes_to_keep=axes_to_keep)
    print("\nSum of redundancy errors for weak invar: ", redundancy_error)

    print("\nIsomorphism bound: ", redundancy_error + entropy_l_given_x_2 * 2)

    print("\n\n==============\n")

    p_L_X_alice = gmm.predict_proba(data)
    p_L_X_bob = gmm2.predict_proba(data)
    p_La_Lb = np.einsum("xa,xb->xab", p_L_X_alice, p_L_X_bob).mean(axis=0)
    p_La_given_lb = p_La_Lb/p_La_Lb.sum(axis=0)
    p_Lb_given_la = p_La_Lb.T / p_La_Lb.T.sum(axis=0)

    entropy_la_given_lb = -(p_La_Lb * jnp.log(p_La_given_lb)).sum() / jnp.log(2)
    entropy_lb_given_la = -(p_La_Lb.T * jnp.log(p_Lb_given_la)).sum() / jnp.log(2)

    print("Entropy L1 | L2: ", entropy_la_given_lb)
    print("Entropy L2 | L1: ", entropy_lb_given_la)

    p_l1 = gmm.predict_proba(data).mean(axis=0)
    p_l2 = gmm2.predict_proba(data).mean(axis=0)

    entropy_l1 = -(p_l1 * jnp.log(p_l1)).sum() / jnp.log(2)
    entropy_l2 = -(p_l2 * jnp.log(p_l2)).sum() / jnp.log(2)
    print("Entropy L1: ", entropy_l1)
    print("Entropy L2: ", entropy_l2)

if __name__ == "__main__":
    main()
