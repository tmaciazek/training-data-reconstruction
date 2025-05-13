import torch
import resource
import argparse
import yaml
import sys, os

import torch.utils.data as data_utils

from torch import nn

from torchvision import transforms
from torchvision.datasets import EMNIST, CIFAR100, CelebA
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, CenterCrop

from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


from torchsummary import summary

import matplotlib.pyplot as plt

from models import VGGNet

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=str, help='Experiment name (MNIST,CIFAR,CelebA).')
parser.add_argument('--model_dir', type=str, default="./models_pretrained",
                    help='Folder for saving the pre-trained NN.')
parser.add_argument('--output_dir', type=str, default="./",
                    help='Output directory.')
parser.add_argument('--config_dir', type=str, default="./config_data",
                    help='Directory containing the config files.')
args = parser.parse_args()
assert args.experiment_id in ['MNIST', 'CIFAR','CelebA'], 'experiment_id must be one of MNIST, CIFAR, CelebA'

os.makedirs(args.model_dir, exist_ok=True)

#sys.stderr = open(os.path.join(args.output_dir,'pretraining_errors.txt'), 'w')

CONF = yaml.load(open(os.path.join(args.config_dir,'pretrain_conf_'+args.experiment_id+'.yml')), Loader=yaml.FullLoader)
print('Experiment ID: ', args.experiment_id)
print(CONF, flush=True)

print("CUDA available?\t", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
        
"""
	Constructing the training and validation data and the classifier NNs
"""

if args.experiment_id == 'MNIST':
	transform_train = transforms.Compose([
		transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=2),
		transforms.ToTensor(),
		transforms.Resize(32, antialias=True),
   		transforms.ConvertImageDtype(torch.float),
   		transforms.Normalize((0.5,), (0.5,)),
	])

	transform_val = transforms.Compose([
		transforms.ToTensor(),
		transforms.Resize(32, antialias=True),
   		transforms.ConvertImageDtype(torch.float),
   		transforms.Normalize((0.5,), (0.5,)),
	])
	
	train_dataset = EMNIST(CONF['DataRoot'], download=False, train = True, split="letters", transform=transform_train)
	val_dataset = EMNIST(CONF['DataRoot'], download=False, train = False, split="letters", transform=transform_val)

	n_data = len(train_dataset.targets)
	n_classes = len(set(train_dataset.targets.numpy()))
	print("No. classes, data size:", n_classes, n_data, flush=True)
	
	classifier = VGGNet(in_channels=1, num_classes=n_classes, hidden_dim=8).to(device)
	
	def weights_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, 0, 0.01)
			nn.init.constant_(m.bias, 0)
	classifier = classifier.apply(weights_init)

elif args.experiment_id == 'CIFAR':
	transform_train = transforms.Compose([
	transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
	transforms.RandomAffine(degrees=15, scale=(0.9, 1.1), shear=2),
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	
	transform_val = transforms.Compose([
	transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	train_dataset = CIFAR100(CONF['DataRoot'], download=False, train = True, transform=transform_train)
	val_dataset = CIFAR100(CONF['DataRoot'], download=False, train = False, transform=transform_val)
	
	n_data = len(train_dataset.targets)
	n_classes = len(set(train_dataset.targets))
	print("No. classes, data size:", n_classes, n_data, flush=True)	
	
	classifier = efficientnet_b0(weights='IMAGENET1K_V1').to(device)

elif args.experiment_id == 'CelebA':
	transform = transforms.Compose([
	transforms.Resize([232], interpolation=InterpolationMode.BICUBIC),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	train_dataset = CelebA(CONF['DataRoot'], download=False, split = 'train', transform=transform)
	val_dataset = CelebA(CONF['DataRoot'], download=False, split = 'valid', transform=transform)
	
	n_data = len(train_dataset)
	n_attr = train_dataset.attr.shape[1]
	print("No. of attributes, data size:", n_attr, n_data, flush=True)	
	
	classifier = wide_resnet50_2(weights='IMAGENET1K_V2').to(device)	

"""
	Classifier head pre-training (only for CIFAR and CELEBA)
"""

# freeze base model parameters and set the BatchNorm layers to .eval()
if args.experiment_id in ['CIFAR', 'CelebA']:
	for param in classifier.parameters():
		param.requires_grad = False
	for m in classifier.modules():
		if isinstance(m, nn.BatchNorm2d): m.eval()
	
	# define data loaders
	train_loader = DataLoader(
    	train_dataset,
    	batch_size=CONF['BATCH_SIZE1'],
    	shuffle=True
    	)
    
	val_loader = DataLoader(
    	val_dataset,
    	batch_size=CONF['BATCH_SIZE1'],
    	shuffle=True
    	)

# replace the output layer
if args.experiment_id == 'CIFAR':
	head_in_features = classifier.classifier[1].in_features
	top = nn.Sequential(
		nn.Dropout(p=0.2, inplace=True),
		nn.Linear(in_features=head_in_features, out_features=100, bias=True)
		)
	top = top.to(device)
	classifier.classifier = top
	
	opt1 = torch.optim.Adam(classifier.classifier.parameters(), lr=CONF['LR1'], weight_decay=CONF['WEIGHT_DECAY1'])
	summary(classifier, input_size=(3, 224, 224))
elif args.experiment_id == 'CelebA':
	head_in_features = classifier.fc.in_features
	top = nn.Sequential(
		nn.Linear(in_features=head_in_features, out_features=40, bias=True),
		nn.Sigmoid()
		)
	top = top.to(device)
	classifier.fc = top
	
	opt1 = torch.optim.Adam(classifier.fc.parameters(), lr=CONF['LR1'], weight_decay=CONF['WEIGHT_DECAY1'])
	summary(classifier, input_size=(3, 224, 224))

# cross-entropy loss for CIFAR, MNIST and BCE for CelebA
if args.experiment_id in ['MNIST','CIFAR']:
	loss_fn = nn.CrossEntropyLoss()
elif args.experiment_id == 'CelebA':
	loss_fn = nn.BCELoss()

# train the top layer
if args.experiment_id in ['CIFAR', 'CelebA']:
	print("Classifier head pre-training...", flush=True)

	n_epochs = CONF['TRAINING_EPOCHS1']
	display_step = CONF['DISPLAY_STEP1']
	i = 0
	running_loss = []	
	running_acc = []
	for epoch in range(n_epochs): 

		for data in train_loader:	
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			if args.experiment_id == 'CelebA':
				 labels = labels.to(torch.float32)
		
			opt1.zero_grad()
		
			outputs = classifier(inputs)
			loss = loss_fn(outputs, labels)
			loss.backward()
			opt1.step()
			running_loss += [loss.item()]
			
			if args.experiment_id == 'CIFAR':
				_, predicted = torch.max(outputs.data, 1)
				running_acc += [100 * (predicted == labels).sum().item()/len(labels)]
			elif args.experiment_id == 'CelebA':
				predicted = outputs > 0.5
				running_acc += [100 * (predicted == labels).sum().item() / (len(labels) * 40)]
				
			if i % display_step == display_step-1:
				train_acc = sum(running_acc[-display_step:]) / display_step
				train_loss = sum(running_loss[-display_step:]) / display_step
				print(f'[{epoch + 1}, {i + 1}] loss: {train_loss}, acc: {train_acc}', flush=True)
			
				correct = 0
				total = 0		
				with torch.no_grad():
					for data in val_loader:
						val_images, val_labels = data
						val_images = val_images.to(device)
						val_labels = val_labels.to(device)
						if args.experiment_id == 'CelebA':
				 			val_labels = val_labels.to(torch.float32)
				 
						val_outputs = classifier(val_images)
						total += val_labels.size(0)
						
						if args.experiment_id == 'CIFAR':
							_, predicted = torch.max(val_outputs.data, 1)
						elif args.experiment_id == 'CelebA':
							predicted = val_outputs > 0.5
						correct += (predicted == val_labels).sum().item()
				if args.experiment_id == 'CIFAR':
					acc = 100 * correct/float(total)
				elif args.experiment_id == 'CelebA':
					acc = 100 * correct/(float(total) * 40)
					
				print(f'Val accuracy: {acc:.3f} %', flush=True)
			
			i = i+1
				
print('\n')

"""
	Entire classifier training/fine-tuning
"""
print("Entire classifier training/fine-tuning...", flush=True)

# unfreeze all the parameters except the BatchNorm layers
for param in classifier.parameters():
    param.requires_grad = True 
for module in classifier.modules():
    if isinstance(module, nn.BatchNorm2d):
    	if hasattr(module, 'weight'):
    		module.weight.requires_grad_(False)
    	if hasattr(module, 'bias'):
    		module.bias.requires_grad_(False)


train_loader = DataLoader(
    train_dataset,
    batch_size=CONF['BATCH_SIZE2'],
    shuffle=True)
    
val_loader = DataLoader(
    val_dataset,
    batch_size=CONF['BATCH_SIZE2'],
    shuffle=True)

opt = torch.optim.Adam(classifier.parameters(), lr=CONF['LR2'])

i = 0
running_loss = []	
running_acc = []
avg_step = 500
for epoch in range(CONF['TRAINING_EPOCHS2']):
	
	classifier.train()
	# set the BatchNorm layers to eval mode
	for m in classifier.modules():
		if isinstance(m, nn.BatchNorm2d):
			m.eval()
			
	for data in train_loader:
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		if args.experiment_id == 'MNIST':
			labels = labels - 1
		elif args.experiment_id == 'CelebA':
			labels = labels.to(torch.float32)
	
		opt.zero_grad()
	
		outputs = classifier(inputs)
		loss = loss_fn(outputs, labels)
		loss.backward()
		opt.step()
		running_loss += [loss.item()]
		
		if args.experiment_id in ['MNIST','CIFAR']:
			_, predicted = torch.max(outputs.data, 1)
			running_acc += [100 * (predicted == labels).sum().item()/len(labels)]
		elif args.experiment_id == 'CelebA':
			predicted = outputs > 0.5
			running_acc += [100 * (predicted == labels).sum().item() / (len(labels) * 40)]
		
		i = i+1
				
	train_acc = sum(running_acc[-avg_step:]) / len(running_acc[-avg_step:])
	train_loss = sum(running_loss[-avg_step:]) / len(running_loss[-avg_step:])
	print(f'[{epoch + 1}, {i + 1}] loss: {train_loss}, acc: {train_acc}', flush=True)
	
	classifier.eval()	
	correct = 0
	total = 0
	with torch.no_grad():
		for data in val_loader:
			val_images, val_labels = data
			val_images = val_images.to(device)
			val_labels = val_labels.to(device)
			if args.experiment_id == 'MNIST':
				val_labels = val_labels - 1
			elif args.experiment_id == 'CelebA':
				 val_labels = val_labels.to(torch.float32) 
			val_outputs = classifier(val_images)
			_, predicted = torch.max(val_outputs.data, 1)
			total += val_labels.size(0)
					
			if args.experiment_id in ['MNIST','CIFAR']:
				_, predicted = torch.max(val_outputs.data, 1)
			elif args.experiment_id == 'CelebA':
				predicted = val_outputs > 0.5
			correct += (predicted == val_labels).sum().item()
						
	if args.experiment_id in ['MNIST','CIFAR']:
		acc = 100 * correct/float(total)
	elif args.experiment_id == 'CelebA':
		acc = 100 * correct/(float(total) * 40)
					
	print(f'Val accuracy: {acc:.3f} %', flush=True)

		
	if epoch % 10 == 9:
		if args.experiment_id == 'MNIST':
			torch.save(classifier.state_dict(), args.model_dir+'/VGGtiny_classifier_EMNIST_'+str(epoch+1)+'EP.pth')
		elif args.experiment_id == 'CIFAR':
			torch.save(classifier.state_dict(), args.model_dir+'/EfficientNetB0_CIFAR100_'+str(epoch+1)+'EP.pth')
		elif args.experiment_id == 'CelebA':
			torch.save(classifier.state_dict(), args.model_dir+'/WideResNet50_CelebA_Attributes_'+str(epoch+1)+'EP.pth')

if args.experiment_id == 'MNIST':
	torch.save(classifier.state_dict(), args.model_dir+'/VGGtiny_classifier_EMNIST_'+str(epoch+1)+'EP.pth')
elif args.experiment_id == 'CIFAR':
	torch.save(classifier.state_dict(), args.model_dir+'/EfficientNetB0_CIFAR100_'+str(epoch+1)+'EP.pth')
elif args.experiment_id == 'CelebA':
	torch.save(classifier.state_dict(), args.model_dir+'/WideResNet50_CelebA_Attributes_'+str(epoch+1)+'EP.pth') 	
