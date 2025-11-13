#GraphGini without GradNorm  
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
from dataloader import *
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.utils import dropout_adj, convert
from scipy.sparse.csgraph import laplacian
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def main():
	#---------------------------------------- Training settings ------------------------------------------------------
	parser = argparse.ArgumentParser()
	parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
	parser.add_argument('--seed', type=int, default=1, help='Random seed.')
	parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
	parser.add_argument('--lr', type=float, default=0.001,help='Initial learning rate.')
	parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
	parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
	parser.add_argument('--heads', type=int, default=1, help='Number of attention heads')
	parser.add_argument('--concat', type=bool, default=False, help='whether use concatenation of multi-head attention')                    
	parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
	parser.add_argument('--negative_slope', type=float, default=0.2, help='Negative slope for leaky relu.')
	parser.add_argument('--alpha', type=float, default=0.0, help='regularization coeff for the individual fairness objective')
	parser.add_argument('--beta', type=float, default=0.0, help='regularization coeff for the GDIF objective')
	parser.add_argument('--dataset', type=str, default='income', choices=['credit', 'pokec_n', 'income'])
	parser.add_argument("--num_layers", type=int, default=1, help="number of hidden layers")
	parser.add_argument('--model', type=str, default='gcn', choices=['gcn','jk', 'gin'])
	parser.add_argument('--initialize_training_epochs', type=int, default=1000, help="number of epochs for backbone GNN")
	args = parser.parse_known_args()[0]
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	# ---------------------------------------------------------- set seeds ----------------------------------------------------------
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.cuda:
	   torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.allow_tf32 = False
	if not args.no_cuda:
	    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	else:
	    device = torch.device('cpu')
	
	modelWeightsFolder = './torch_weights/'
	bestinitEncoderWeightsName = f"{modelWeightsFolder}/best_initEncoder_weights.pt"
	bestEncoderWeightsName = f"{modelWeightsFolder}/best_bestEncoder_weights.pt"
	#------------------------------------------------ Data Loading ------------------------------------------------------------------
	adj, features, labels, idx_train, idx_val, idx_test, sens = load_data(args.dataset)
	#------------------------------------- Getting Similarity matrix and Laplacian---------------------------------------------------
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
	# lap_1 = lap_list[0]
	# lap_2 = lap_list[1] 
	# print(lap.shape, lap_1.shape, lap_2.shape)
	# print(type(lap))
	# from scipy.sparse.linalg import eigsh
	# eigenvalues_lap, eigenvectors_lap = eigsh(lap)
	# eigenvalues_lap_1, eigenvectors_lap_1 = eigsh(lap_1)
	# eigenvalues_lap_2, eigenvectors_lap_2 = eigsh(lap_2)
	# print("lap", len(eigenvalues_lap), eigenvalues_lap)
	# print("lap_1",len(eigenvalues_lap_1), eigenvalues_lap_1)
	# print("lap_2",len(eigenvalues_lap_2), eigenvalues_lap_2)
	# import matplotlib.pyplot as plt
	# plt.scatter(range(len(eigenvalues_lap)),eigenvalues_lap, color='green', label='lap')
	# plt.scatter(range(len(eigenvalues_lap_1)),eigenvalues_lap_1, color='blue', label='lap1')
	# plt.scatter(range(len(eigenvalues_lap_2)),eigenvalues_lap_2, color='red', label='lap2')
	# plt.legend()
	# plt.xlabel("Eigenvalue number")
	# plt.ylabel("Laplacian Eigenvalues")
	# plt.title(args.dataset)
	# plt.show()
	#####################################################################3
	lap = convert_sparse_matrix_to_sparse_tensor(lap)
	lap_list = [convert_sparse_matrix_to_sparse_tensor(X) for X in lap_list]
	lap_1 = lap_list[0]
	lap_2 = lap_list[1]
	m_u1 = m_list[0]
	m_u2 = m_list[1]
	
	##############################
	filterSigma = 1.6
	print("**********************Filter Similarity Matrix**********************")
	print("Sigma:", filterSigma)
	sim1 = calculate_similarity_matrix_with_th(adj, features, filterSigma)
	sim_edge_index1, sim_edge_weight1 = convert.from_scipy_sparse_matrix(sim1)
	sim_edge_weight1 = sim_edge_weight1.type(torch.FloatTensor)
	sim_edge_index1 = sim_edge_index1.to(device)
	sim_edge_weight1 = sim_edge_weight1.to(device)
	#------------------------------------------ Model and optimizer -----------------------------------------------------------------
	num_class = 1 #labels.unique().shape[0]-1
	if args.model == 'gcn':
	    model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=num_class, dropout=args.dropout)
	    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif args.model == 'gin':
	    model = GIN(nfeat=features.shape[1], nhid=args.hidden, nclass=num_class, dropout=args.dropout)
	    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	elif args.model == 'jk':
		model = JK(nfeat=features.shape[1], nhid=args.hidden, nclass=num_class, dropout=args.dropout)
		optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	model = model.to(device)
	
	ifgModel = gdEncoder(num_layers = args.num_layers, nfeat=args.hidden, nhid=args.hidden, nclass=num_class, heads=args.heads, negative_slope=args.negative_slope, concat=args.concat, dropout=args.dropout)
	ifgOptimizer = optim.Adam(ifgModel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	ifgModel = ifgModel.to(device)
	#----------------------------------------------------------------------------------------------------------------------------------
	#                                                     Train  initial model
	#----------------------------------------------------------------------------------------------------------------------------------
	t_total = time.time()
	best_perf_val = 0
	best_total_loss_val = np.inf
	features = features.to(device)
	edge_index = edge_index.to(device)
	labels = labels.to(device)
	sim_edge_index = sim_edge_index.to(device)
	sim_edge_weight = sim_edge_weight.to(device)
	lap = lap.to(device)
	lap_1 = lap_1.to(device)
	lap_2 = lap_2.to(device)
	print(f"---------------------------------------Embedding Initialization-------------------------------------------")
	for epoch in range(args.initialize_training_epochs+1):
	    t = time.time()
	    model.train()
	    optimizer.zero_grad()
	    output = model(features, edge_index)
	    loss_label_init_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
	    auc_roc_init_train = roc_auc_score(labels.cpu().numpy()[idx_train], output.detach().cpu().numpy()[idx_train])
	    individual_unfairness_vanilla = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap,output))).item()
	    f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_1, output)))/m_u1
	    f_u1 = f_u1.item()
	
	    f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_2, output)))/m_u2
	    f_u2 = f_u2.item()
	    GD_vanilla = max(f_u1/f_u2, f_u2/f_u1)
	    loss_label_init_train.backward()
	    optimizer.step()
	    if epoch % 100 == 0:
	        print(f"[Train] Epoch {epoch}: ")
	        print(f"Embedding Initialize: loss_label_train: {loss_label_init_train.item():.4f}, auc_roc_train: {auc_roc_init_train:.4f}, individual_unfairness_vanilla: {individual_unfairness_vanilla:.4f}, GD_vanilla {GD_vanilla:.4f}")
	#-------------------------------------------------------------------------------------------------------------------------------------
	#                                                     Train Regularized GNN
	#------------------------------------------------------------------------------------------------------------------------------------
	iters = 0
	layer = ifgModel.fc
	lr2 = 5e-4
	alpha2 = 0.12
	log_weights = []
	log_loss = []
	log = True
	print(f"---------------------------------------Training Regularized GNN---------------------------------------------")
	for epoch in range(args.epochs+1):
	    t = time.time()
	    ifgModel.train()
	    ifgOptimizer.zero_grad()
	    with torch.no_grad():
	        output = model.body(features, edge_index)
	    ifgOutput = ifgModel(output, sim_edge_index1, sim_edge_weight1)
	    loss_label_ifg_train = F.binary_cross_entropy_with_logits(ifgOutput[idx_train], labels[idx_train].unsqueeze(1).float().to(device))
	    auc_roc_ifg_train = roc_auc_score(labels.cpu().numpy()[idx_train], ifgOutput.detach().cpu().numpy()[idx_train])
	
	    ifair_loss = torch.trace( torch.mm( ifgOutput.t(), torch.sparse.mm(lap, ifgOutput) ))
	    f_u1 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_1, ifgOutput)))/m_u1
	    f_u2 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_2, ifgOutput)))/m_u2
	    GD = max(f_u1/f_u2, f_u2/f_u1)
	    ifg_loss = -(f_u1/f_u2-1)*(f_u2/f_u1-1)
	    loss_ifg_train = loss_label_ifg_train + args.alpha * ifair_loss + args.beta * ifg_loss
	    loss_ifg_train.backward()
	    ifgOptimizer.step()
	    ######################## validation ###########################################
	    #Validation: Evaluate validation set performance separately
	    ifgModel.eval()
	    ifgOutput = ifgModel(output, sim_edge_index1, sim_edge_weight1)
	    loss_label_ifg_val = F.binary_cross_entropy_with_logits(ifgOutput[idx_val], labels[idx_val].unsqueeze(1).float().to(device))
	    individual_unfairness = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap, ifgOutput))).item()
	    f_u1 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_1, ifgOutput)))/m_u1
	    f_u1 = f_u1.item()
	    f_u2 = torch.trace(torch.mm(ifgOutput.t(), torch.sparse.mm(lap_2, ifgOutput)))/m_u2
	    f_u2 = f_u2.item()
	    GD = max(f_u1/f_u2, f_u2/f_u1)
	    ifg_loss = -(f_u1/f_u2-1)*(f_u2/f_u1-1)
	  
	    # printing metrics
	    preds_ifg = (ifgOutput.squeeze()>0).type_as(labels)
	    auc_roc_ifg_val = roc_auc_score(labels.cpu().numpy()[idx_val], ifgOutput.detach().cpu().numpy()[idx_val])
	    total_loss_val = loss_label_ifg_val + args.alpha * individual_unfairness + args.beta * ifg_loss
	    if (total_loss_val < best_total_loss_val) and (epoch > 500):
	        best_total_loss_val = total_loss_val
	        torch.save(ifgModel.state_dict(), bestEncoderWeightsName)
	    if epoch % 100 == 0:
	        vec = [f_u1, f_u2]
	        #print("vec", vec)
	        print(f"[Train] Epoch {epoch}:")
	        print(f"loss_label_train {loss_label_ifg_train.item():.4f},loss_label_val {total_loss_val:.4f}, auc_roc_train: {auc_roc_ifg_train.item():.4f}, , auc_roc_val: {auc_roc_ifg_val:.4f}")
	
	torch.save(model.state_dict(), bestinitEncoderWeightsName)
	#-----------------------------------------------------------------------------------------------------------------------------
	#                                                     Testing
	#--------------------------------------------------------------------------------------------------------------------------
	model.load_state_dict(torch.load(bestinitEncoderWeightsName))
	model.eval()
	output = model.body(features, edge_index)
	ifgModel.load_state_dict(torch.load(bestEncoderWeightsName))
	ifgModel.eval()
	output, attention_weights = ifgModel(output, sim_edge_index1.to(device), sim_edge_weight1.to(device), return_attention_weights=True )
	attention_weights = torch.sparse_coo_tensor(attention_weights[0], attention_weights[1])
	attention_weights = attention_weights.detach()
	attention_weights = attention_weights.coalesce()
	output_preds = (output.squeeze()>0).type_as(labels)
	auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output.detach().cpu().numpy()[idx_test.cpu()])
	individual_unfairness = torch.trace( torch.mm( output.t(), torch.sparse.mm(lap, output) ) ).item()
	f_u1 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_1, output)))/m_u1
	f_u1 = f_u1.item()
	f_u2 = torch.trace(torch.mm(output.t(), torch.sparse.mm(lap_2, output)))/m_u2
	f_u2 = f_u2.item()
	if_group_pct_diff = np.abs(f_u1-f_u2)/min(f_u1, f_u2)
	GD = max(f_u1/f_u2, f_u2/f_u1)
	
	print("---Testing---")
	print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
	print(f'Total Individual Unfairness: {individual_unfairness}')
	print(f'Individual Unfairness for Group 1: {f_u1}')
	print(f'Individual Unfairness for Group 2: {f_u2}')
	print(f'GD: {GD}')
	#############################################################################
	numCluster = 6
	df = features[idx_test].cpu()
	output_prob = torch.sigmoid(output)
	
	def gini_np(x):
	    total = 0
	    for i, xi in enumerate(x[:-1], 1):
	        total += np.sum(np.abs(xi - x[i:]))
	    return total / (len(x)**2 * np.mean(x))
	
	
	kmeanModel = KMeans(n_clusters=numCluster,random_state=0)
	kmeanModel.fit(df)
	labels_kmeans = kmeanModel.labels_ 
	print("cluster, class0Gini, Class1Gini, AllClassGini")
	for i in range(numCluster):
	    ##print(df[labels_kmeans==i].shape)
	    output_cluster = output_prob.detach().cpu().numpy()[idx_test.cpu()][labels_kmeans==i]
	    labels_cluster = labels.cpu().numpy()[idx_test.cpu()][labels_kmeans==i]
	    #print(output_cluster.shape, labels_cluster.shape)
	    idx = np.where([labels_cluster==0])[1]
	    output_gini_torch_0 = output_cluster[idx] #- np.min(output_cluster[idx])
	    #print(gini_np(output_gini_torch_0))
	    idx = np.where([labels_cluster==1])[1]
	    output_gini_torch_1 = output_cluster[idx] #- np.min(output_cluster[idx])
	
	    print(i,", ", gini_np(output_gini_torch_0),", ",gini_np(output_gini_torch_1), ", ",gini_np(output_cluster))
	    #print(i, gini_np(output_gini_torch_0),gini_np(output_gini_torch_1), gini_np(output_cluster))
	##########################################################################################################################  
	output_vanilla = model(features, edge_index)
	output_vanilla_prob = torch.sigmoid(output_vanilla)
	auc_roc_test_vanilla = roc_auc_score(labels.cpu().numpy()[idx_test.cpu()], output_vanilla.detach().cpu().numpy()[idx_test.cpu()])
	
	individual_unfairness_vanilla = torch.trace( torch.mm( output_vanilla.t(), torch.sparse.mm(lap, output_vanilla) ) ).item()
	f_u1 = torch.trace(torch.mm(output_vanilla.t(), torch.sparse.mm(lap_1, output_vanilla)))/m_u1
	f_u1 = f_u1.item()
	f_u2 = torch.trace(torch.mm(output_vanilla.t(), torch.sparse.mm(lap_2, output_vanilla)))/m_u2
	f_u2 = f_u2.item()
	if_group_pct_diff = np.abs(f_u1-f_u2)/min(f_u1, f_u2)
	GDIF = max(f_u1/f_u2, f_u2/f_u1)
	print("---Testing on vanilla---")
	print("The AUCROC of estimator: {:.4f}".format(auc_roc_test_vanilla))
	print(f'Total Individual Unfairness: {individual_unfairness_vanilla}')
	print(f'Individual Unfairness for Group 1: {f_u1}')
	print(f'Individual Unfairness for Group 2: {f_u2}')
	print(f'GDIF: {GDIF}')
	
	print("cluster, class0Gini, Class1Gini, AllClassGini")
	for i in range(numCluster):
	    ##print(df[labels_kmeans==i].shape)
	    output_cluster = output_vanilla_prob.detach().cpu().numpy()[idx_test.cpu()][labels_kmeans==i]
	    labels_cluster = labels.cpu().numpy()[idx_test.cpu()][labels_kmeans==i]
	    #print(output_cluster.shape, labels_cluster.shape)
	    idx = np.where([labels_cluster==0])[1]
	    output_gini_torch_0 = output_cluster[idx] #- np.min(output_cluster[idx])
	    #print(gini_np(output_gini_torch_0))
	    idx = np.where([labels_cluster==1])[1]
	    output_gini_torch_1 = output_cluster[idx] #- np.min(output_cluster[idx])
	    print(i,", ", gini_np(output_gini_torch_0),", ",gini_np(output_gini_torch_1), ", ",gini_np(output_cluster))



if __name__ == "__main__":
	main()
