import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

import sklearn as sk
import math
import random
import itertools
import operator
from collections import Counter

from tqdm import tqdm

from torchvision import transforms
from torchvision import datasets
from ipywidgets import IntProgress

import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.autograd import Function
from torch.utils.tensorboard import SummaryWriter

from collections import Iterable

import time
import datetime
import logging
import pickle
import json

import time, argparse

from sklearn.metrics import roc_auc_score

def apply_damethod(Xs, Xt, ys=None, yt=None, 
                    dataset_name=None, src_name=None, 
                    tgt_name=None, batch_size=32, train_pct=0.8,
                    epochs=50, device='cuda:0'):

    """ Apply damethod.

    Parameters
    ----------
    Xs : np.array of shape (n_samples, n_features), optional (default=None)
        The source instances.
    Xt : np.array of shape (n_samples, n_features), optional (default=None)
        The target instances.
    ys : np.array of shape (n_samples,), optional (default=None)
        The ground truth of the source instances.
    yt : np.array of shape (n_samples,), optional (default=None)
        The ground truth of the target instances.
    dataset_name (example): 'mnist_a10' / 'mnist_to_mnist_m_a10'
    src_name (example): 'mnist_source_n1_a1_v0'
    tgt_name (example): 'mnist_v0'
    batch_size: 32, batch size for DataLoaders
    train_pct: 0.8, splitting source data to 0.8 (train), 0.2 (validation)
    epochs: 50, training epochs
    device: cuda:0, GPU or CPU

    Returns
    -------
    auc, correct.item()/total

    auc: roc_auc_score of predictions on the target data
    correct.item()/total: accuracy of predictions on the target data
    """

    seed_everything(42)

    if torch.cuda.is_available():  
        dev = device
        device = torch.device(dev)
    
    #device = 'cpu'

    dimensions_dict = {'mnist_a10': 28, 'mnist_m_a10': 28, 'mnist_lesser_a10': 28, 'usps_a10': 16, 
                        'svhn_a10': 32, 'synthdigits_a10': 32, 'visda17_train_combined_a10': 224,
                        'mnist_to_mnist_m_a10': 28, 'mnist_to_mnist_m_a10_debug': 28, 'mnist_to_svhn_a10': 32, 
                        'mnist_to_usps_a10': 28, 'svhn_to_mnist_a10': 32, 
                        'synthdigits_to_svhn_a10': 32, 'usps_to_mnist_a10': 28}
    
    h_w = dimensions_dict[dataset_name]

    tensorboard_dir = '/home/tanwl/LocIT/damethod_tensorboard_amended/' + dataset_name + '/' + src_name + '_to_' + tgt_name
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    source_data = input_dataset(X=Xs, y=ys, h_w=h_w, transform=True)
    #source_data = input_dataset('/home/tanwl/LocIT/data/mnist_a10/source/mnist_source_n1_a1_v0.csv')

    train_size = int(train_pct * len(source_data))
    val_size = len(source_data) - train_size
    train_data, val_data = torch.utils.data.random_split(source_data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    test_data = input_dataset(X=Xt, y=yt, h_w=h_w, transform=True)
    #test_data = input_dataset('/home/tanwl/LocIT/data/mnist_a10/target/mnist_v0.csv')
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print('Initializing Models')
    # Initialize Models
    feature_extractor = ResNet50Fc()
    feature_extractor.to(device)
    
    classifier = CLS(input_dim=2048, output_dim=2, hidden_layer_dim=256)
    classifier.to(device)
    
    discriminator_adv = AdversarialNetwork(256)
    discriminator_adv.to(device)

    discriminator_non_adv = AdversarialNetwork(256)
    discriminator_non_adv.to(device)

    # Set weights for CrossEntropyLoss criterion due to imbalanced dataset
    weights = torch.tensor([90.0, 10.0], dtype=torch.float32) # 90% normal, 10% anomalies
    weights = weights / weights.sum()
    weights = 1.0 / weights
    weights = weights / weights.sum()

    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none', weight=weights) # THIS IS NOW VALID SINCE CROSSENTROPY DOES SOFTMAX FIRST THEN NLLLOSS
    criterion_ce_none = nn.BCELoss(reduction='none')
    criterion_ce_mean = nn.BCELoss(reduction='mean')

    if train_size // batch_size > 0:
        max_iter = int(epochs * (train_size // batch_size))
    else:
        max_iter = 1000
    
    scheduler = lambda step, initial_lr: inverseDecayScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
    optimizer_finetune = OptimWithScheduler(
        optim.SGD(feature_extractor.parameters(), lr=0.01 / 10.0, weight_decay=0.0005, momentum=0.9, nesterov=True),
        scheduler)

    optimizer_classifier = OptimWithScheduler(
        optim.SGD(classifier.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.9, nesterov=True),
        scheduler)

    optimizer_discriminator_adv = OptimWithScheduler(
        optim.SGD(discriminator_adv.parameters(), lr=0.005, weight_decay=0.0005, momentum=0.9, nesterov=True),
        scheduler)

    optimizer_discriminator_non_adv = OptimWithScheduler(
        optim.SGD(discriminator_non_adv.parameters(), lr=0.005, weight_decay=0.0005, momentum=0.9, nesterov=True),
        scheduler)

    total_loss_lst = []
    classifier_loss_lst = []
    disc_adv_loss_lst = []
    disc_non_adv_loss_lst = []

    classifier_train_accuracy_lst = []
    classifier_val_accuracy_lst = []
    discriminator_adv_accuracy_lst = []
    classifier_test_accuracy_lst = []
    discriminator_non_adv_accuracy_lst = []

    global_step = 0

    for epoch in range(1, epochs+1):
        print('Epoch: ' + str(epoch))
        
        # Training
        feature_extractor.train()
        classifier.train()
        discriminator_adv.train()
        discriminator_non_adv.train()
        
        len_dataloader = min(len(train_dataloader), len(test_dataloader))

        for step, ((source_samples, source_labels), (target_samples, target_labels)) in tqdm(enumerate(zip(train_dataloader, test_dataloader)), position=0, leave=True):
        
            source_labels_gpu = source_labels.to(device)
            source_samples_gpu = source_samples.to(device)
            
            target_labels_gpu = target_labels.to(device)
            target_samples_gpu = target_samples.to(device)
            
            # Calculate alpha weight for ReverseLayerF
            p = float(step + epoch * len_dataloader) / epochs + 1 / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # Feature Extractor, F
            source_out = feature_extractor(source_samples_gpu)
            target_out = feature_extractor(target_samples_gpu)
            
            # Classifier, G Loss
            source_input_features, source_hidden_layer_features, _, source_logits, source_probs = classifier(source_out)
            target_input_features, target_hidden_layer_features, _, target_logits, target_probs = classifier(target_out)
            
            classifier_loss = criterion(source_probs, source_labels_gpu.long())
            classifier_loss = torch.mean(classifier_loss, dim=0, keepdim=True)   
            
            # Gradient Reversal
            source_hidden_layer_features_reversed = ReverseLayerF.apply(source_hidden_layer_features, alpha)
            target_hidden_layer_features_reversed = ReverseLayerF.apply(target_hidden_layer_features, alpha)

            # Adversarial Discriminator, D
            domain_prob_discriminator_adv_source = discriminator_adv(source_hidden_layer_features_reversed)
            domain_prob_discriminator_adv_target = discriminator_adv(target_hidden_layer_features_reversed)

            # Non-Adversarial Discriminator, D'
            domain_prob_discriminator_non_adv_source = discriminator_non_adv(source_hidden_layer_features.detach())
            domain_prob_discriminator_non_adv_target = discriminator_non_adv(target_hidden_layer_features.detach())        

            # Calculate w_s
            source_share_weight = get_source_share_weight(domain_prob_discriminator_non_adv_source, source_logits, domain_temperature=1.0, class_temperature=10.0)
            source_share_weight = normalize_weight(source_share_weight)

            # Calculate w_t
            target_share_weight = get_target_share_weight(domain_prob_discriminator_non_adv_target, target_logits, domain_temperature=1.0, class_temperature=1.0)
            target_share_weight = normalize_weight(target_share_weight)
            
            # Compute D and D' Loss
            adv_loss = torch.zeros(1, 1).to(device)
            non_adv_loss = torch.zeros(1, 1).to(device)
            
            #source_share_weight = 1.0
            #target_share_weight = 1.0
            
            # Compute D (Adversarial) Loss
            # Source Domain Label is 1
            #domain_label_source = torch.ones(len(source_samples)).long()
            #domain_label_target = torch.zeros(len(target_samples)).long()
            domain_label_source = torch.ones_like(domain_prob_discriminator_adv_source).to(device)
            domain_label_target = torch.zeros_like(domain_prob_discriminator_adv_target).to(device)
            
            # tmp = source_share_weight * criterion_ce_none(domain_prob_discriminator_adv_source, torch.ones_like(domain_prob_discriminator_adv_source))
            #tmp = criterion_ce_none(domain_prob_discriminator_adv_source, torch.ones_like(domain_prob_discriminator_adv_source).long()))
            tmp = source_share_weight * criterion_ce_none(domain_prob_discriminator_adv_source, domain_label_source)
            adv_loss = adv_loss + torch.mean(tmp, dim=0, keepdim=True)
            
            # Target Domain Label is 0
            # tmp = target_share_weight * criterion_ce_none(domain_prob_discriminator_adv_target, torch.zeros_like(domain_prob_discriminator_adv_target))
            #tmp = criterion_ce_none(domain_prob_discriminator_adv_target, torch.zeros_like(domain_prob_discriminator_adv_target).long()))
            tmp = target_share_weight * criterion_ce_none(domain_prob_discriminator_adv_target, domain_label_target)
            adv_loss = adv_loss + torch.mean(tmp, dim=0, keepdim=True)

            # Compute D' (Non-Adversarial) Loss
            # Source Domain Label is 1

            non_adv_loss = non_adv_loss + criterion_ce_mean(domain_prob_discriminator_non_adv_source, domain_label_source)
            #non_adv_loss = non_adv_loss + criterion_ce_mean(domain_prob_discriminator_non_adv_source, torch.ones_like(domain_prob_discriminator_non_adv_source).long()))
            # Target Domain Label is 0
            non_adv_loss = non_adv_loss + criterion_ce_mean(domain_prob_discriminator_non_adv_target, domain_label_target)
            #non_adv_loss = non_adv_loss + criterion_ce_mean(domain_prob_discriminator_non_adv_target, torch.zeros_like(domain_prob_discriminator_non_adv_target).long()))
            
            with OptimizerManager(
                    [optimizer_finetune, optimizer_classifier, optimizer_discriminator_adv, optimizer_discriminator_non_adv]):
                total_loss = classifier_loss + adv_loss + non_adv_loss
                total_loss.backward()
                
            global_step = global_step + 1
        
        writer.add_scalar('classifier_loss', classifier_loss, epoch)
        writer.add_scalar('disc_adv_loss', adv_loss, epoch)
        writer.add_scalar('disc_non_adv_loss', non_adv_loss, epoch)
        writer.add_scalar('total_loss', total_loss, epoch)
        
        #print('classifier_loss: ' + str(classifier_loss.item()))
        #print('disc_adv_loss: ' + str(adv_loss.item()))
        #print('disc_non_adv_loss: ' + str(non_adv_loss.item()))
        #print('total_loss: ' + str(total_loss.item()))
        
        classifier_loss_lst.append(classifier_loss.item())
        disc_adv_loss_lst.append(adv_loss.item())
        disc_non_adv_loss_lst.append(non_adv_loss.item())
        total_loss_lst.append(total_loss.item())
        
        # Calculating Training Accuracy for this Epoch
        correct = 0
        total = len(train_data)
        feature_extractor.eval()
        classifier.eval()

        with torch.no_grad():
            for samples, labels in tqdm(train_dataloader, position=0, leave=True):
                labels_gpu = labels.to(device)
                samples_gpu = samples.to(device)

                out = feature_extractor(samples_gpu)
                input_features, hidden_layer_features, _, logits, probs = classifier(out)

                predictions = torch.argmax(probs, dim=1)
                correct += torch.sum((predictions == labels_gpu).float())

        print('Training Accuracy: {}'.format(correct.item()/total))
        classifier_train_accuracy_lst.append(correct.item()/total)
        writer.add_scalar('classifier_train_accuracy', correct.item()/total, epoch)

        # Calculating Validation Accuracy for this Epoch
        correct = 0
        total = len(val_data)
        feature_extractor.eval()
        classifier.eval()

        with torch.no_grad():
            for samples, labels in tqdm(val_dataloader, position=0, leave=True):
                labels_gpu = labels.to(device)
                samples_gpu = samples.to(device)
 
                out = feature_extractor(samples_gpu)
                input_features, hidden_layer_features, _, logits, probs = classifier(out)

                predictions = torch.argmax(probs, dim=1)
                correct += torch.sum((predictions == labels_gpu).float())

        print('Validation Accuracy: {}'.format(correct.item()/total))
        classifier_val_accuracy_lst.append(correct.item()/total)
        writer.add_scalar('classifier_val_accuracy', correct.item()/total, epoch)
        
        # Calculating Accuracy (of Adversarial Discriminator) for this Epoch
        correct = 0
        total = len(train_data) + len(test_data)
        feature_extractor.eval()
        classifier.eval()
        discriminator_adv.eval()
        
        with torch.no_grad():
            for samples, labels in tqdm(train_dataloader, position=0, leave=True):
                samples_gpu = samples.to(device)
   
                out = feature_extractor(samples_gpu)
                input_features, hidden_layer_features, _, logits, probs = classifier(out)
                
                domain_prob_discriminator_adv_source = discriminator_adv(hidden_layer_features) # (32, 1)
                #print('domain_prob_discriminator_adv_source: ' + str(domain_prob_discriminator_adv_source))
                domain_label_source = torch.ones_like(domain_prob_discriminator_adv_source).to(device)
                #print('domain_label_source: ' + str(domain_label_source)) # (32, 1) -
                predictions = (domain_prob_discriminator_adv_source > 0.5).float()
                #print('predictions: ' + str(predictions))
                
                correct += torch.sum((predictions == domain_label_source).float())
                #print('correct: ' + str(correct))
            
            for samples, labels in tqdm(test_dataloader, position=0, leave=True):
                samples_gpu = samples.to(device)

                out = feature_extractor(samples_gpu)
                input_features, hidden_layer_features, _, logits, probs = classifier(out)
                
                domain_prob_discriminator_adv_target = discriminator_adv(hidden_layer_features) # (32, 1)
                #print('domain_prob_discriminator_adv_target: ' + str(domain_prob_discriminator_adv_target))
                domain_label_target = torch.zeros_like(domain_prob_discriminator_adv_target).to(device)
                #print('domain_label_target: ' + str(domain_label_target)) # (32, 1) -
                predictions = (domain_prob_discriminator_adv_target < 0.5).float()
                #print('predictions: ' + str(predictions))
                
                correct += torch.sum((predictions == domain_label_target).float())
                #print('correct: ' + str(correct))
                
        print('Accuracy (of Adversarial Discriminator): {}'.format(correct.item()/total))
        discriminator_adv_accuracy_lst.append(correct.item()/total)
        writer.add_scalar('discriminator_adv_accuracy', correct.item()/total, epoch)

        # Calculating Accuracy (of Non-Adversarial Discriminator) for this Epoch
        correct = 0
        total = len(train_data) + len(test_data)
        feature_extractor.eval()
        classifier.eval()
        discriminator_adv.eval()
        
        with torch.no_grad():
            for samples, labels in tqdm(train_dataloader, position=0, leave=True):
                samples_gpu = samples.to(device)

                out = feature_extractor(samples_gpu)
                input_features, hidden_layer_features, _, logits, probs = classifier(out)
                
                domain_prob_discriminator_non_adv_source = discriminator_non_adv(hidden_layer_features)
                domain_label_source = torch.ones_like(domain_prob_discriminator_non_adv_source).to(device)
                predictions = (domain_prob_discriminator_non_adv_source > 0.5).float()
                
                correct += torch.sum((predictions == domain_label_source).float())

            for samples, labels in tqdm(test_dataloader, position=0, leave=True):
                samples_gpu = samples.to(device)

                out = feature_extractor(samples_gpu)
                input_features, hidden_layer_features, _, logits, probs = classifier(out)
                
                domain_prob_discriminator_non_adv_target = discriminator_non_adv(hidden_layer_features)
                domain_label_target = torch.zeros_like(domain_prob_discriminator_non_adv_target).to(device)
                predictions = (domain_prob_discriminator_non_adv_target < 0.5).float()
                
                correct += torch.sum((predictions == domain_label_target).float())

        print('Accuracy (of Non-Adversarial Discriminator): {}'.format(correct.item()/total))
        discriminator_non_adv_accuracy_lst.append(correct.item()/total)
        writer.add_scalar('discriminator_non_adv_accuracy', correct.item()/total, epoch)
        
    print('Training Complete!')
    print('Saving Models')
    if src_name[-2:] == 'v0':
        torch.save(feature_extractor.state_dict(), tensorboard_dir + '/feature_extractor.pt')
        torch.save(classifier.state_dict(), tensorboard_dir + '/classifier.pt')
        torch.save(discriminator_adv.state_dict(), tensorboard_dir + '/discriminator_adv.pt')
        torch.save(discriminator_non_adv.state_dict(), tensorboard_dir + '/discriminator_non_adv.pt')
    print('Done!')

    print('Saving Accuracies & Losses')
    with open(tensorboard_dir + '/total_loss_lst.json', 'w') as f:
        json.dump(total_loss_lst, f)
    with open(tensorboard_dir + '/classifier_loss_lst.json', 'w') as f:
        json.dump(classifier_loss_lst, f)
    with open(tensorboard_dir + '/disc_adv_loss_lst.json', 'w') as f:
        json.dump(disc_adv_loss_lst, f)
    with open(tensorboard_dir + '/disc_non_adv_loss_lst.json', 'w') as f:
        json.dump(disc_non_adv_loss_lst, f)
    with open(tensorboard_dir + '/classifier_train_accuracy_lst.json', 'w') as f:
        json.dump(classifier_train_accuracy_lst, f)
    with open(tensorboard_dir + '/classifier_val_accuracy_lst.json', 'w') as f:
        json.dump(classifier_val_accuracy_lst, f)
    with open(tensorboard_dir + '/discriminator_adv_accuracy_lst.json', 'w') as f:
        json.dump(discriminator_adv_accuracy_lst, f)
    with open(tensorboard_dir + '/discriminator_non_adv_accuracy_lst.json', 'w') as f:
        json.dump(discriminator_non_adv_accuracy_lst, f)
    print('Done!')

    # Output AUROC directly
    correct = 0
    total = len(test_data)
    feature_extractor.eval()
    classifier.eval()
    test_dataloader_final = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    with torch.no_grad():
    # Iterate through test set
        for samples, labels in test_dataloader_final:
            labels_gpu = labels.to(device)
            samples_gpu = samples.to(device)

            out = feature_extractor(samples_gpu)
            input_features, hidden_layer_features, _, logits, probs = classifier(out)

            predictions = torch.argmax(probs, dim=1)
            #print('predictions: ' + str(predictions))
            #print('labels_gpu.detach().cpu().numpy()' + str(labels_gpu.detach().cpu().numpy()))
            #print('probs.detach().cpu().numpy()' + str(probs.detach().cpu().numpy()))
            #print('probs.detach().cpu().numpy()[:, 1]' + str(probs.detach().cpu().numpy()[:, 1]))
            auc = roc_auc_score(y_true=labels_gpu.detach().cpu().numpy(), y_score=probs.detach().cpu().numpy()[:, 1])
            correct += torch.sum((predictions == labels_gpu).float())

    print('Test AUROC: {}'.format(auc))
    print('Test Accuracy: {}'.format(correct.item()/total))
    
    return auc, correct.item()/total


# Misc Functions & Classes
def seed_everything(seed=42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

# Numpy Version
class input_dataset(Dataset):
    def __init__(self, X, y, h_w, transform=True):
        self.X = X # numpy array
        self.y = y # numpy array
        self.h_w = h_w
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        h_w = self.h_w
        X = self.X[index]
        y = self.y[index]
        X = torch.tensor(X)
        X = X.view(-1, h_w, h_w) # X_tensor is now (1, 28, 28)
        if self.transform:
            X = self.transform(X)
        return X, y
    
    def __len__(self):
        return self.len

# BaseFeatureExtractor to freeze Batch Norm statistics during training

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # Freeze Batch Norm statistics during training
        # model.children() is a generator that returns layers of the model from which you can extract your parameter tensors
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)

# ResNet50Fc inherits from BaseFeatureExtractor

class ResNet50Fc(BaseFeatureExtractor):
    """
    NOTE: INPUT IMAGE, X, NEEDS TO BE NORMALIZED FIRST TO FIT ResNet50, i.e.,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    
    # Input: (Batch Size, C, H, W) = (Batch Size, 3, 224, 224)
    # Output: (Batch Size, Features) = (Batch Size, 2048)
    """
    
    # model_path was '/workspace/fubo/resnet50.pth' in original paper
    def __init__(self,model_path=None):
        super(ResNet50Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('Invalid model directory!')
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class CLS(nn.Module):
    """
    Two-layer MLP for binary classification of features coming from F
    # Input: (Batch Size, Features) = (Batch Size, 2048)
    """
    def __init__(self, input_dim, output_dim, hidden_layer_dim=256):
        super(CLS, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_layer_dim)
        self.fc = nn.Linear(hidden_layer_dim, output_dim)
        self.main = nn.Sequential(self.hidden_layer, nn.ReLU(inplace=True), self.fc, nn.Softmax(dim=-1))
        
    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out

class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with GRL following DANN
    """
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.main(x)
        return y

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class OptimizerManager:
    """
    automatic call op.zero_grad() when enter, call op.step() when exit
    usage::
        with OptimizerManager(op): # or with OptimizerManager([op1, op2])
            b = net.forward(a)
            b.backward(torch.ones_like(b))
    """
    def __init__(self, optims):
        self.optims = optims if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None # release reference, to avoid imexplicit reference
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True

class OptimWithScheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1

def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    '''
    change as initial_lr * (1 + gamma * min(1.0, iter / max_iter) ) ** (- power)
    as known as inv learning rate sheduler in caffe,
    see https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto

    the default gamma and power come from <Domain-Adversarial Training of Neural Networks>

    code to see how it changes(decays to %20 at %10 * max_iter under default arg)::

        from matplotlib import pyplot as plt

        ys = [inverseDecaySheduler(x, 1e-3) for x in range(10000)]
        xs = [x for x in range(10000)]

        plt.plot(xs, ys)
        plt.show()

    '''
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))

# domain_out is domain_prob_discriminator_non_adv_source / domain_prob_discriminator_non_adv_target  (32, 1)
# before_softmax is source_logits / target_logits (32, 2)
def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

def get_source_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    before_softmax = before_softmax / class_temperature # before_softmax is (32, 2)
    after_softmax = nn.Softmax(-1)(before_softmax) # after_softmax is (32, 2)
    domain_logit = reverse_sigmoid(domain_out) # domain_logit is (32, 1)
    domain_logit = domain_logit / domain_temperature
    domain_out = nn.Sigmoid()(domain_logit) # domain_out is (32, 1)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True) # entropy is (32, 1)
    entropy_norm = entropy / np.log(after_softmax.size(1)) # entropy_norm is (32, 1), after_softmax.size(1) is 2
    weight = entropy_norm - domain_out # weight is (32, 1)
    weight = weight.detach()
    return weight # weight is (32, 1)


def get_target_share_weight(domain_out, before_softmax, domain_temperature=1.0, class_temperature=10.0):
    return - get_source_share_weight(domain_out, before_softmax, domain_temperature, class_temperature)

# x is (32, 1)
def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    x = x / torch.mean(x)
    return x.detach()