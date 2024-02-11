import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from datetime import datetime
import argparse
import pickle
from utils import *
from models import *
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from scipy.sparse.csgraph import laplacian
#---------------------------------------- Training settings ------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='income', choices=['credit', 'pokec_n', 'income'])
args = parser.parse_known_args()[0]
#---------------------------------------- Load data --------------------------------------------------
if args.dataset == 'credit':
	sens_attr = "Age"  # column number after feature process is 1
	sens_idx = 1
	predict_attr = 'NoDefaultNextMonth'
	label_number = 6000
	path_credit = "./dataset/credit"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_credit(args.dataset, sens_attr, predict_attr, path=path_credit,label_number=label_number)
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features
elif args.dataset == 'income':
	sens_attr = "race"  # column number after feature process is 1
	sens_idx = 8
	predict_attr = 'income'
	label_number = 6000
	path_income = "./dataset/income"
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_income(args.dataset, sens_attr, predict_attr, path=path_income,label_number=label_number)
	norm_features = feature_norm(features)
	norm_features[:, sens_idx] = features[:, sens_idx]
	features = norm_features
# test on this dataset later ############################3   
elif args.dataset.split('_')[0] == 'pokec':
    if args.dataset == 'pokec_z':
        args.dataset = 'region_job'
    elif args.dataset == 'pokec_n':
        args.dataset = 'region_job_2'
    sens_attr = "AGE"
    predict_attr = "I_am_working_in_field"
    label_number = 10000
    sens_idx = 4
    path="./dataset/pokec/"
    adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec(args.dataset,sens_attr, predict_attr, path=path, label_number=label_number)

######################################
else:
	print('Invalid dataset name!!')
	exit(0)
#------------------------------------- Getting Similarity matrix and Laplacian--------------------------------------------------------------------------
print(f"Getting similarity matrix...")
edge_index = convert.from_scipy_sparse_matrix(adj)[0]
sim = calculate_similarity_matrix(adj, features, metric='cosine')
sim_edge_index, sim_edge_weight = convert.from_scipy_sparse_matrix(sim)
sim_edge_weight = sim_edge_weight.type(torch.FloatTensor)
lap = laplacian(sim)
print(f"Similarity matrix nonzero entries: {torch.count_nonzero(sim_edge_weight)}")

print("Get laplacians for Individual Fairness of groups calculations...")
try:
    with open('./stored_laplacians/' + args.dataset + '.pickle', 'rb') as f:
        loadLaplacians = pickle.load(f)
    lap_list, m_list, avgSimD_list = loadLaplacians['lap_list'], loadLaplacians['m_list'], loadLaplacians['avgSimD_list']
    print("Laplacians loaded from previous runs")
except FileNotFoundError:
    print("Calculating laplacians...(this may take a while for pokec_n)")
    lap_list, m_list, avgSimD_list = calculate_group_lap(sim, sens)
    saveLaplacians = {}
    saveLaplacians['lap_list'] = lap_list
    saveLaplacians['m_list'] = m_list
    saveLaplacians['avgSimD_list'] = avgSimD_list
    with open('./stored_laplacians/' + args.dataset + '.pickle', 'wb') as f:
        pickle.dump(saveLaplacians, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Laplacians calculated and stored.")
######################### Laplacian Spectra #############################################  
lap_1 = lap_list[0]
lap_2 = lap_list[1] 
print(lap.shape, lap_1.shape, lap_2.shape)
print(type(lap))
from scipy.sparse.linalg import eigsh
eigenvalues_lap, eigenvectors_lap = eigsh(lap)
eigenvalues_lap_1, eigenvectors_lap_1 = eigsh(lap_1)
eigenvalues_lap_2, eigenvectors_lap_2 = eigsh(lap_2)
print("lap", len(eigenvalues_lap), eigenvalues_lap)
print("lap_1",len(eigenvalues_lap_1), eigenvalues_lap_1)
print("lap_2",len(eigenvalues_lap_2), eigenvalues_lap_2)
import matplotlib.pyplot as plt
plt.scatter(range(len(eigenvalues_lap)),eigenvalues_lap, color='green', label='lap')
plt.scatter(range(len(eigenvalues_lap_1)),eigenvalues_lap_1, color='blue', label='lap1')
plt.scatter(range(len(eigenvalues_lap_2)),eigenvalues_lap_2, color='red', label='lap2')
plt.legend()
plt.xlabel("Eigenvalue number")
plt.ylabel("Laplacian Eigenvalues")
plt.title(args.dataset)
plt.show()