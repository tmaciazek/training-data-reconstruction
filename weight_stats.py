import torch
import resource
import sys
import numpy as np
import argparse
import sys, os

print("CUDA available?\t", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
	
parser = argparse.ArgumentParser()
parser.add_argument('--filename_header', type=str, help='The header of the weight data filenames.')
parser.add_argument('--seed_range', type=int, help='Number of the weight data permutation seeds.')
parser.add_argument('--shadow_dir', type=str, default="./shadow_models_data",
                    help='Directory containing the shadow models.')
parser.add_argument('--stats_dir', type=str, default="./weight_stats_data",
                    help='Directory containing the shadow models.')

args = parser.parse_args()                    
os.makedirs(args.stats_dir, exist_ok=True)

print(args.filename_header)
train_seeds = [i for i in range(args.seed_range)]

flag = True
missing = []
found_seeds = []
for seed in train_seeds:
	fname = args.shadow_dir+'/'+args.filename_header+'_train_seed_'+str(seed)+'.npy'
	flag = flag * os.path.isfile(fname)
	if os.path.isfile(fname)==False: 
		missing.append(seed)
	else:
		found_seeds.append(seed)
assert flag, 'Missing seeds: '+str(missing)

w_sum = 0.
w_sq_sum = 0.
n_per_seed = 0.
wTw_sum = 0.
n = 0
for seed in train_seeds:
	print("Loading training data seed ", seed, flush=True)
	train_weights_seed = np.load(args.shadow_dir+'/'+args.filename_header+'_train_seed_'+str(seed)+'.npy').astype(np.float64)
	train_weights_seed = torch.from_numpy(train_weights_seed).to(device)
	n += train_weights_seed.shape[0]
	w_sum_seed = torch.sum(train_weights_seed, axis=0).reshape(1, -1)
	w_sq_sum_seed = torch.sum(torch.square(train_weights_seed), axis=0).reshape(1, -1)
	w_sum += w_sum_seed
	w_sq_sum += w_sq_sum_seed
	wTw_sum += train_weights_seed.T @ train_weights_seed
	

w_std_pre = n * w_sq_sum - torch.square(w_sum)
assert torch.sum(w_std_pre < 0)==0, 'Negative variance! Not enough precision?'
w_std = torch.sqrt(w_std_pre)/n
w_mean = w_sum/n
w_cov = (n * wTw_sum - w_sum.T @ w_sum)/(n*n)


np.save(args.stats_dir+'/'+'stats_mean_' + args.filename_header + '.npy', w_mean.cpu().numpy())
np.save(args.stats_dir+'/'+'stats_std_' + args.filename_header + '.npy', w_std.cpu().numpy())

#print(tuple(w_mean.shape), tuple(w_std.shape), torch.sum(w_std==0.0).item(), torch.min(w_std).item())

assert torch.sum(w_std == 0.0)==0, 'Vanishing variance!'
w_cov = torch.divide(w_cov, w_std.T @ w_std)
np.save(args.stats_dir+'/'+'stats_covnorm_' + args.filename_header + '.npy', w_cov.cpu().numpy())



