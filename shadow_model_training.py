import torch
import resource
import time
import argparse
import yaml
import sys, os

import torch.utils.data as data_utils

from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryAccuracy

import matplotlib.pyplot as plt

from torchsummary import summary

from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_id', type=str, help='Dtaset name (MNIST, CIFAR10, CIFAR100, CelebA).')
parser.add_argument('--split', type=str, default='train', help='Generate train/test shadow models?')
parser.add_argument('--permutation_seed', type=int, help='Random seed used to permute the data.')
parser.add_argument('--models_per_seed', type=int, help='How many shadow models per seed? See documentation.')
parser.add_argument('--binary', type=str, default="F",
                    help='Training binary classifiers? Answer T/F')
parser.add_argument('--config_dir', type=str, default="./config_data",
                    help='Directory containing the config files.')
parser.add_argument('--features_dir', type=str, default="./deep_features_data",
                    help='Directory containing the deep features.')
parser.add_argument('--shadow_dir', type=str, default="./shadow_models_data",
                    help='Directory for saving the shadow models.')
parser.add_argument('--filename_appendix', type=str, default="classifier",
                    help='Shadow model file name appendix')
                    
args = parser.parse_args()
assert args.data_id in ['MNIST', 'CIFAR10', 'CIFAR100','CelebA'], 'data_id must be one of MNIST, CIFAR10, CIFAR100, CelebA'
assert args.binary in ['F', 'T'], 'binary must be one of T/F'
assert args.split in ['test', 'train'], 'split must be one of train/test'

os.makedirs(args.shadow_dir, exist_ok=True)

flag_binary = (args.binary=='T')

if args.data_id in ['CIFAR10', 'CIFAR100']:
	if flag_binary:	
		CONF = yaml.load(open(os.path.join(args.config_dir,'shadow_conf_binary_CIFAR.yml')), Loader=yaml.FullLoader)
	else:
		CONF = yaml.load(open(os.path.join(args.config_dir,'shadow_conf_multiclass_CIFAR.yml')), Loader=yaml.FullLoader)
elif args.data_id == 'CelebA':
	CONF = yaml.load(open(os.path.join(args.config_dir,'shadow_conf_CelebA.yml')), Loader=yaml.FullLoader)
elif flag_binary:
	CONF = yaml.load(open(os.path.join(args.config_dir,'shadow_conf_binary_'+args.data_id+'.yml')), Loader=yaml.FullLoader)
else:
	CONF = yaml.load(open(os.path.join(args.config_dir,'shadow_conf_multiclass_'+args.data_id+'.yml')), Loader=yaml.FullLoader)

assert (CONF['TRAINING_EPOCHS'] == 'Auto') or type(CONF['TRAINING_EPOCHS']) == int, 'TRAINING_EPOCHS must be Auto or an integer'
assert (CONF['BATCH_SIZE'] == 'Full') or type(CONF['BATCH_SIZE']) == int, 'BATCH_SIZE must be Full or an integer'
assert CONF['OPTIMIZER'] in ['SGD', 'Adam'], 'OPTIMIZER must be one of SGD/Adam'

print('Data ID: ', args.data_id)
print('Shadow model split: ', args.split)
if args.data_id == 'CelebA':
	print('Training binary classifiers?', 'True')
else:
	print('Training binary classifiers?', flag_binary)

print(CONF, flush=True)

"""
	Load features data
"""

features_train = torch.from_numpy(np.load(args.features_dir+"/"+args.data_id+"_features_train.npy"))
labels_train = torch.from_numpy(np.load(args.features_dir+"/"+args.data_id+"_labels_train.npy"))
features_val = torch.from_numpy(np.load(args.features_dir+"/"+args.data_id+"_features_val.npy"))
labels_val = torch.from_numpy(np.load(args.features_dir+"/"+args.data_id+"_labels_val.npy"))

"""
	Prepare the head NN
"""

features_dim = features_train.shape[1]

if args.data_id in ['CIFAR10', 'CIFAR100', 'MNIST']:
	if flag_binary:
		head = nn.Sequential(
			nn.Linear(in_features=features_dim, out_features=8, bias=True),
			nn.ReLU(),
			nn.Linear(in_features=8, out_features=1, bias=True)
		)
	else:
		head = nn.Sequential(
  			nn.Linear(in_features=features_dim, out_features=10, bias=True)
		)
elif args.data_id == 'CelebA':
	head = nn.Sequential(
			nn.Linear(in_features=features_dim, out_features=4, bias=True),
			nn.ReLU(),
			nn.Linear(in_features=4, out_features=1, bias=True)
		)

print(summary(head,(features_dim,)))

"""
	Classifier pre-training
"""

class_size = CONF['CLASS_SIZE']
seed = args.permutation_seed
models_per_seed = args.models_per_seed
if CONF['TRAINING_EPOCHS'] == 'Auto':
	if args.data_id == 'MNIST':
		n_epochs = 26 + 6 * class_size
	elif args.data_id in ['CIFAR10', 'CIFAR100']:
		n_epochs = 38 + 10 * class_size
	print("Training epochs no.:\t", n_epochs, flush=True)
else:
	n_epochs = CONF['TRAINING_EPOCHS']
lr =  CONF['LR']

if CONF['BATCH_SIZE'] == 'Full':
	batch_size = class_size * 10
else: 
	batch_size = CONF['BATCH_SIZE']
init_std = CONF['INIT_STD']

def weights_init(m):
	if isinstance(m, nn.Linear):
		nn.init.normal_(m.weight, 0, init_std) #0.002
		nn.init.constant_(m.bias, 0)

#prepare shadow model training sets for CIFAR/MNIST
if args.data_id in ['MNIST','CIFAR10', 'CIFAR100']:
	if args.split == 'train':
		training_sets, label_sets = get_balanced_sets(features_train, labels_train, class_size=class_size, seed=seed)
	else:
		training_sets, label_sets = get_balanced_sets(features_val, labels_val, class_size=class_size, seed=seed)
	print("Total no. of shadow models available:\t", len(training_sets), flush=True)
	assert len(training_sets) >= models_per_seed
elif args.data_id == 'CelebA':	
	rng = np.random.default_rng(seed=seed)
	if args.split == 'train':
		data, labels = features_train, labels_train
	else:
		data, labels = features_val, labels_val
	sample_inds = torch.from_numpy(rng.permutation(range(len(data))).reshape(-1,1))
	data = torch.take_along_dim(data, sample_inds, dim=0)
	labels = torch.take_along_dim(labels, sample_inds.reshape(-1,), dim=0)

# for binary classification change labels accordingly
if flag_binary:
	if args.data_id == 'MNIST':
		label_sets = (label_sets % 2).to(float)
		labels_val = (labels_val % 2).to(float).reshape(-1,1)
	elif args.data_id in ['CIFAR10', 'CIFAR100']:
		label_vehicle = torch.from_numpy(np.array([0,1,8,9], dtype=int))
		label_sets = torch.isin(label_sets,label_vehicle).to(float)
		labels_val = torch.isin(labels_val,label_vehicle).to(float).reshape(-1,1)

# define the loss function
if args.data_id == 'CelebA':
	pos_weight = torch.tensor([CONF['POS_WEIGHT']])
	loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
elif flag_binary:
	loss_fn = nn.BCEWithLogitsLoss()
else: 
	loss_fn = nn.CrossEntropyLoss()


opt_name = CONF['OPTIMIZER']

"""
	Shadow head training loop
"""

if args.data_id in ['MNIST','CIFAR10', 'CIFAR100']:
	trained_models_weights = []
	tot_time = 0.0
	running_acc = []
	for training_no in range(models_per_seed):
		print("Seed, training_no:\t" + str(seed)+"\t"+str(training_no)+"/"+str(models_per_seed), flush=True)
		tic = time.perf_counter()
		head = head.apply(weights_init)
		if opt_name == 'SGD':
			opt = torch.optim.SGD(head.parameters(), lr=lr, weight_decay = CONF['WEIGHT_DECAY'])
		else:
			opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay = CONF['WEIGHT_DECAY'])	
	
		X_train = training_sets[training_no]
		labels = label_sets[training_no]
	
		dataset = data_utils.TensorDataset(X_train, labels)
		dataloader = DataLoader(dataset, batch_size=batch_size)

		for epoch in range(n_epochs):
			for X, label in dataloader:
				opt.zero_grad()
				outputs = head(X)
				if flag_binary:
					loss = loss_fn(outputs, label.reshape(-1,1))
				else:
					loss = loss_fn(outputs, label.reshape(-1,))
				loss.backward()
				opt.step()
	
		with torch.no_grad():
			val_outputs = head(features_val)
			if flag_binary:
				predicted = (val_outputs > 0).to(float)
			else:
				_, predicted = torch.max(val_outputs.data, 1)
			acc = 100*(predicted == labels_val).sum().item()/float(len(val_outputs))
			print("Final training loss, final val acc:", loss.item(), acc)
			running_acc.append(acc)
	
		weights_flat = torch.cat([p.data.flatten() for p in head.parameters()], 0).numpy()
		trained_models_weights.append(weights_flat)
	
		toc = time.perf_counter()
		tot_time += toc - tic

	
	print(
    	"Data MEM usage:\t"
    	+ str(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)))
    	+ " GB", flush=True
	)

	print("Training time per shadow model [s]:\t", tot_time/models_per_seed)
	print("Total training time [min]:\t", tot_time/60)
	print("Mean, std of val accuracy:\t", np.mean(running_acc), np.std(running_acc))

elif args.data_id == 'CelebA':

	acc_list = []
	tp_list = []
	tn_list = []
	trained_models_weights = []
	tot_time = 0
	it = 0
	loss_threshold = CONF['LOSS_THRESHOLD']
	while len(trained_models_weights) < models_per_seed:
		tic = time.perf_counter()
		mask = (labels == labels[it])
		
		# only select ids with more than class_size photos
		if torch.sum(mask) > class_size:
			train_ind = torch.nonzero(mask).reshape(-1,)
			features_id = data[mask]
			features_comp = data[~mask]
			
			# select the train and validation sets
			nV = len(features_id)-class_size # size of the validation pos class
			
			#randomly choose the neg class images
			rand_inds_train = torch.randint(high=len(features_comp), size=(9*class_size, 1))
			rand_inds_val = torch.randint(high=len(features_comp), size=(9 * nV,1))
			train_rest = torch.take_along_dim(features_comp, rand_inds_train, dim=0)
			val_rest = torch.take_along_dim(features_comp, rand_inds_val, dim=0)
			
			#concat the train and val sets
			X_train = torch.cat([features_id[:class_size], train_rest])
			Y_train = torch.cat([torch.ones(class_size),torch.zeros(9*class_size)],0).to(torch.float32).reshape(-1,1)

			X_val = features_id[class_size:]
			X_val = torch.cat([X_val, val_rest])
			Y_val = torch.cat([torch.ones(nV),torch.zeros(9*nV)],0).to(torch.float32).reshape(-1,1)

			if opt_name == 'SGD':
				opt = torch.optim.SGD(head.parameters(), lr=lr, weight_decay = CONF['WEIGHT_DECAY'])
			else:
				opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay = CONF['WEIGHT_DECAY'])

			dataset = data_utils.TensorDataset(X_train, Y_train)
			dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

			class_weights = torch.tensor([10.])
			loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
			acc_metric = BinaryAccuracy()
		
			flag_stop = False
			attempts = 0
			print('Training set ', it)
			
			# Restart training until loss_threshold is achieved.
			# Otherwise, training sometimes gets stuck in a local minimum
			# and produces a bad quality classifier.
			while not flag_stop:
				head = head.apply(weights_init)
				for epoch in range(n_epochs):
					for X, lab in dataloader:
						opt.zero_grad()
						outputs = head(X)
						loss = loss_fn(outputs, lab)
						loss.backward()
						opt.step()
					#print(epoch, loss.item())
			
				acc_metric.reset()
				with torch.no_grad():
					val_preds = torch.sigmoid(head(X_val))
					train_preds = torch.sigmoid(head(X_train))
					acc_metric.update(val_preds.reshape(-1, ), Y_val.reshape(-1, ))
					acc = acc_metric.compute().item()

				acc_metric.reset()
				acc_metric.update(val_preds.reshape(-1, )[:nV], Y_val.reshape(-1, )[:nV])
				tp = acc_metric.compute().item()
				acc_metric.reset()
				acc_metric.update(val_preds.reshape(-1, )[nV:], Y_val.reshape(-1, )[nV:])
				tn = acc_metric.compute().item()
				acc_metric.reset()
				acc_metric.update(train_preds.reshape(-1, ), Y_train.reshape(-1, ))
				acc_metric.reset()
				print('Attempt ', attempts+1, ' final loss: ', loss.item(), flush=True)
				attempts += 1
				flag_stop = ((loss.item() < loss_threshold) or (attempts > 100))
			
				if flag_stop:
					acc_list += [acc]
					tp_list += [tp]
					tn_list += [tn]
	
			toc = time.perf_counter()
			tot_time += toc - tic
			print("Training time [s]:\t", toc-tic, flush=True)
			print('\n')
		
			weights_flat = torch.cat([p.data.flatten() for p in head.parameters()], 0).numpy()
			trained_models_weights.append(weights_flat)
	
		it+=1



	acc_list = np.array(acc_list)
	tp_list = np.array(tp_list)
	tn_list = np.array(tn_list)
	print("Training time per shadow model [s]:\t", tot_time/models_per_seed)
	print("Total training time [min]:\t", tot_time/60)
	print('ACC mean, STD:', np.mean(acc_list),np.std(acc_list))
	print('TPR mean, STD:', np.mean(tp_list),np.std(tp_list))
	print('TNR mean, STD:', np.mean(tn_list),np.std(tn_list))


if flag_binary:
	app = args.filename_appendix + '_binary'
else:
	app = args.filename_appendix

if args.split == 'train':
	filename = args.data_id +'_'+app+'_'+'N'+str(10*class_size)+'_train_seed_'+str(seed)
else:
	filename = args.data_id+'_'+app+'_'+'N'+str(10*class_size)+'_test_seed_'+str(seed)

np.save(args.shadow_dir +'/'+filename+".npy", np.array(trained_models_weights))
