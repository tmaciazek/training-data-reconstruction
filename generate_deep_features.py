import numpy as np
import torch
import resource
import argparse
import yaml
import sys, os
import time


from torch import nn

import torch.utils.data as data_utils

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, CelebA
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.transforms import InterpolationMode, CenterCrop

from torchvision.models import wide_resnet50_2
from torchvision.models import efficientnet_b0

from models import VGGNet

from torchsummary import summary

print("CUDA available?\t", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
	
parser = argparse.ArgumentParser()
parser.add_argument('--data_id', type=str, help='Dtaset name (MNIST, CIFAR10, CIFAR100, CelebA).')
parser.add_argument('--model', type=str, help='Pre-trained model file name.')
parser.add_argument('--model_dir', type=str, default="./models_pretrained",
                    help='Folder containing the pre-trained NNs.')
parser.add_argument('--config_dir', type=str, default="./config_data",
                    help='Directory containing the config files.')
parser.add_argument('--features_dir', type=str, default="./deep_features_data",
                    help='Directory for saving the deep features.')
parser.add_argument('--celeba_img_dir', type=str, default="./celeba_img64",
                    help='Directory for saving the resized CelebA images.')
args = parser.parse_args()
assert args.data_id in ['MNIST', 'CIFAR10', 'CIFAR100','CelebA'], 'data_id must be one of MNIST, CIFAR10, CIFAR100, CelebA'
assert args.model[-4:] == '.pth', 'Model file name must end with .pth'
if args.data_id in ['CIFAR10', 'CIFAR100']:
	assert args.model[:12] == 'EfficientNet', 'CIFAR data requires EfficientNetB0'
elif args.data_id == 'MNIST':
	assert args.model[:3] == 'VGG', 'MNIST data requires VGG'
elif args.data_id == 'CelebA':
	assert args.model[:10] == 'WideResNet', 'CelebA data requires WideResNet50'

os.makedirs(args.features_dir, exist_ok=True)
if args.data_id == 'CelebA':
	os.makedirs(args.celeba_img_dir, exist_ok=True)

if args.data_id in ['CIFAR10', 'CIFAR100']:	
	CONF = yaml.load(open(os.path.join(args.config_dir,'pretrain_conf_CIFAR.yml')), Loader=yaml.FullLoader)
else:
	CONF = yaml.load(open(os.path.join(args.config_dir,'pretrain_conf_'+args.data_id+'.yml')), Loader=yaml.FullLoader)
print('Data ID: ', args.data_id)
        
"""
	Constructing the training and test data
"""

if args.data_id == 'MNIST':
	img_shape = (32, 32)
	transform = transforms.Compose([
		transforms.Resize(32, antialias=True),
		transforms.ToTensor(),
   		transforms.ConvertImageDtype(torch.float),
   		transforms.Normalize((0.5,), (0.5,)),
   		])

	train_dataset = MNIST(CONF['DataRoot'], download=False, train = True, transform=transform)
	val_dataset = MNIST(CONF['DataRoot'], download=False, train = False, transform=transform)

	n_train_data = len(train_dataset.targets)
	n_val_data = len(val_dataset.targets)
	n_classes = len(set(train_dataset.targets.numpy()))
	print("No. classes, train data size, val data size:", n_classes, n_train_data, n_val_data, flush=True)
	
	classifier = VGGNet(in_channels=1, num_classes=26, hidden_dim=4).to(device)
	classifier.load_state_dict(torch.load(args.model_dir+'/'+args.model, map_location=device))
	
	# remove the output neurons
	classifier.fc_top = nn.Identity()	

elif args.data_id in ['CIFAR10', 'CIFAR100']:

	transform = transforms.Compose([
		transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	
	if args.data_id == 'CIFAR100':
		train_dataset = CIFAR100(CONF['DataRoot'], download=False, train = True, transform=transform)
		val_dataset = CIFAR100(CONF['DataRoot'], download=False, train = False, transform=transform)
	elif args.data_id == 'CIFAR10':
		train_dataset = CIFAR10(CONF['DataRoot'], download=False, train = True, transform=transform)
		val_dataset = CIFAR10(CONF['DataRoot'], download=False, train = False, transform=transform)
	
	n_train_data = len(train_dataset.targets)
	n_val_data = len(val_dataset.targets)
	n_classes = len(set(train_dataset.targets))
	print("No. classes, train data size, val data size:", n_classes, n_train_data, n_val_data, flush=True)	
	
	classifier = efficientnet_b0(weights=None).to(device)
	head_in_features = classifier.classifier[1].in_features
	top = nn.Sequential(
		nn.Dropout(p=0.2, inplace=True),
		nn.Linear(in_features=head_in_features, out_features=100, bias=True)
		)
	top = top.to(device)
	classifier.classifier = top
	classifier.load_state_dict(torch.load(args.model_dir+'/'+args.model, map_location=device))
	classifier.classifier = nn.Identity()

elif args.data_id == 'CelebA':
	transform = transforms.Compose([
		transforms.Resize([232], interpolation=InterpolationMode.BICUBIC),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	train_dataset = CelebA(CONF['DataRoot'], download=False, split = 'train', target_type='identity', transform=transform)
	val_dataset = CelebA(CONF['DataRoot'], download=False, split = 'valid', target_type='identity', transform=transform)
	# join train_dataset and val_dataset into bigger train_dataset
	train_dataset = ConcatDataset([train_dataset, val_dataset])
	# the test split forms the val_dataset
	val_dataset = CelebA(CONF['DataRoot'], download=False, split = 'test', target_type='identity', transform=transform)
	
	n_train_data = len(train_dataset)
	n_val_data = len(val_dataset)
	print("Train data size, val data size:", n_train_data, n_val_data, flush=True)	
	
	classifier = wide_resnet50_2(weights=None).to(device)	
	top = nn.Sequential(
  		nn.Linear(in_features=2048, out_features=40, bias=True),
  		nn.Sigmoid()
		)
	top = top.to(device)
	classifier.fc = top
	classifier.load_state_dict(torch.load(args.model_dir+'/'+args.model, map_location=device))
	classifier.fc = nn.Identity()

"""
	Extract the deep features
"""

# set the classifier to eval mode
classifier.eval()

batch_size = 512
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False)
    
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False)

tic = time.perf_counter()

features_train = []
labels_train = []
with torch.no_grad():
	for i, data in enumerate(train_loader):
		print('Train batch ', i, '/', len(train_loader), flush=True)
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		features = classifier(inputs)
		features_train.append(features)
		labels_train.append(labels)
features_train = torch.cat(features_train, axis=0)
labels_train = torch.cat(labels_train, axis=0)
print('Train features, label shape: ', features_train.shape, labels_train.shape, flush=True)

features_val = []
labels_val = []
with torch.no_grad():
	for i, data in enumerate(val_loader):
		print('Val batch ', i, '/', len(val_loader), flush=True)
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		features = classifier(inputs)
		features_val.append(features)
		labels_val.append(labels)
features_val = torch.cat(features_val, axis=0)
labels_val = torch.cat(labels_val, axis=0)
print('Val features, label shape: ', features_val.shape, labels_val.shape, flush=True)

toc = time.perf_counter()
prec_time = toc-tic
print(f'Features extraction took: {prec_time:.3f} seconds', flush=True)

np.save(args.features_dir+"/"+args.data_id+"_features_train.npy", features_train.cpu().numpy())
np.save(args.features_dir+"/"+args.data_id+"_labels_train.npy", labels_train.cpu().numpy())
np.save(args.features_dir+"/"+args.data_id+"_features_val.npy", features_val.cpu().numpy())
np.save(args.features_dir+"/"+args.data_id+"_labels_val.npy", labels_val.cpu().numpy())

'''
	For CelebA only - resize the images in the dataset
'''

if args.data_id == 'CelebA':
	print('Generating resized images.', flush=True)
	tic = time.perf_counter()

	transform = transforms.Compose([
		transforms.Resize([232], interpolation=InterpolationMode.BICUBIC),
		transforms.CenterCrop(224),
		transforms.Resize([64], interpolation=InterpolationMode.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.5], std=[0.5])
	])

	train_dataset = CelebA('.', download=False, split = 'train', target_type='identity', transform=transform)
	val_dataset = CelebA('.', download=False, split = 'valid', target_type='identity', transform=transform)	
	train_dataset = ConcatDataset([train_dataset, val_dataset])
	val_dataset = CelebA('.', download=False, split = 'test', target_type='identity', transform=transform)

	batch_size = 2048
	train_loader = DataLoader(
    	train_dataset,
    	batch_size=batch_size,
    	shuffle=False)
    
	val_loader = DataLoader(
    	val_dataset,
    	batch_size=batch_size,
    	shuffle=False)
    	
	img_train = []
	for i, data in enumerate(train_loader):
		print('Train batch ', i, '/', len(train_loader), flush=True)
		img, _ = data
		img_train.append(img)
	img_train = torch.cat(img_train, 0)
	print(img_train.shape)
	np.save(args.celeba_img_dir+"/CelebA_img_64x64_train.npy", img_train.cpu().numpy())
	
	img_val = []
	for i, data in enumerate(val_loader):
		print('Val batch ', i, '/', len(val_loader), flush=True)
		img, _ = data
		img_val.append(img)
	img_val = torch.cat(img_val, 0)
	print(img_val.shape)
	np.save(args.celeba_img_dir+"/CelebA_img_64x64_val.npy", img_val.cpu().numpy())
	
	toc = time.perf_counter()
	prec_time = toc-tic
	print(f'Image resizing took: {prec_time:.3f} seconds', flush=True)
    	
    