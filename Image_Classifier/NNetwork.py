import numpy as np
import os, random, json, time

import torch
from torch import nn, optim 
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from Utilities import Utils
#-------------------------------------------------------------------
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.5
EPOCHS = 8
GPU = False

class NNetwork():
    def __init__(self, gpu=False):
        self._gpu = gpu
        self.epochs = EPOCHS
        self.dropout_prob = DROPOUT_PROB
        self.learning_rate = LEARNING_RATE
        self.model = None
        self.optimizer = None
        self.arch = None
        self.cat_to_name = None
    #-------------------------------------------------------------------
    @property
    def gpu(self):
        return torch.cuda.is_available() and self._gpu  
    #-------------------------------------------------------------------
    @gpu.setter
    def gpu(self, value):
        self._gpu = value
    #-------------------------------------------------------------------       
    @Utils.tryit()
    def get_classifier_input(self):
        #print(self.model.classifier)
        if(hasattr(self.model, 'classifier')): 
            if(hasattr(self.model.classifier, 'in_features')):
                return self.model.classifier.in_features
            for c in self.model.classifier:
                if(hasattr(c, 'in_features')):
                    return c.in_features
                if(hasattr(c,'in_channels')):
                    return c.in_channels
        if(hasattr(self.model, 'fc') and hasattr(self.model.fc, 'in_features')):
            return self.model.fc.in_features
    #-------------------------------------------------------------------
    @Utils.tryit()
    def build_model(self, arch, hidden_units, classifier_output, learning_rate=LEARNING_RATE, dropout_prob=DROPOUT_PROB):
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.classifier_output = classifier_output
        
        self.model = getattr(models, arch)(pretrained=True)
        
        self.classifier_input = self.get_classifier_input()
        
        for param in self.model.parameters():
            param.requires_grad = False

        hidden_units = [hidden_units] if(type(hidden_units) != list) else hidden_units
        self.model.classifier = self.build_classifier(self.classifier_input, hidden_units, classifier_output, dropout_prob)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate )    
        
        return self.model
    #-------------------------------------------------------------------
    @Utils.tryit()
    def build_classifier(self, classifier_input, hidden_layers, classifier_output, dropout_prob):
        classifier = nn.Sequential()
        self.input_layer = nn.Linear(classifier_input, hidden_layers[0])
        self.all_layers = [self.input_layer]
        all_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers = [nn.Linear(features_in, features_out) for features_in, features_out in all_layers]
        self.all_layers.extend(self.hidden_layers)
        self.output_layer = nn.Linear(hidden_layers[-1], classifier_output)
        self.all_layers.append(self.output_layer)
        
        for i, hl in enumerate(self.all_layers):
            classifier.add_module('fc{}'.format(i), hl)
            if(i < len(self.all_layers)-1):
                classifier.add_module('drop{}'.format(i), nn.Dropout(dropout_prob))
                classifier.add_module('relu{}'.format(i), nn.ReLU())
            else:
                classifier.add_module('output', nn.LogSoftmax(dim=1)) 
              
        return classifier
    #-------------------------------------------------------------------    
    @Utils.tryit()
    def train_model(self, epochs, train_dataloader, valid_dataloader=None, gpu=GPU):
        self.epochs, self.gpu = epochs, gpu
        steps = running_loss = accuracy = 0     
        dataloaders = {'train': train_dataloader, 'valid': valid_dataloader}

        self.model.cuda() if(self.gpu) else self.model.cpu()

        start = time.time()
        Utils.log(0,'Training started')

        for epoch in range(epochs):
            for mode,dataloader in dataloaders.items():
                if(mode == 'test' or dataloader == None): 
                    continue

                self.model.train() if(mode=='train') else self.model.eval()

                pass_count = 0

                for data in dataloader:
                    pass_count += 1
                    inputs, labels = data

                    if(self.gpu):
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    self.optimizer.zero_grad()

                    output = self.model.forward(inputs) 
                    loss = self.criterion(output, labels)

                    if mode == 'train':
                        loss.backward()
                        self.optimizer.step()                

                    running_loss += loss.item()
                    ps = torch.exp(output).data
                    equality = (labels.data == ps.max(1)[1])

                    float_tensor = torch.cuda.FloatTensor() if(self.gpu) else torch.FloatTensor()
                    accuracy = equality.type_as(float_tensor).mean()

                if mode == 'train':
                    Utils.log(1, "Epoch: {}/{} ", epoch + 1, epochs)
                    Utils.log(2, "Training Loss: {:.4f}  ", running_loss/pass_count)
                          
                else:
                    Utils.log(2, "Validation Loss: {:.4f} - Validation Accuracy: {:.4f}", running_loss/pass_count, accuracy)

                running_loss = 0

        time_elapsed = time.time() - start
        Utils.log(0, "Total time: {:.0f}m {:.0f}s", time_elapsed//60, time_elapsed % 60)
    #-------------------------------------------------------------------    
    @Utils.tryit()
    def test_model_accuracy(self, dataloader):
        self.model.eval()

        accuracy = pass_count = 0

        self.model.cuda() if(self.gpu) else self.model.cpu()

        for data in dataloader:
            pass_count += 1
            images, labels = data
            images, labels = [Variable(images.cuda()), Variable(labels.cuda())] if(self.gpu) \
            else [Variable(images), Variable(labels)]

            output = self.model.forward(images)
            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        Utils.log(0,"Accuracy Test Result: {:.4f}", accuracy/pass_count)
    #-------------------------------------------------------------------    
    @Utils.tryit()    
    def predict(self, image, topk=5, gpu=False):
        self.gpu = gpu
        self.model.cuda() if(self.gpu) else self.model.cpu()
        self.model.eval()

        image = torch.from_numpy(np.array([image]))
        image = image.float()
        image = image.cuda() if(self.gpu) else Variable(image)
       
        with torch.no_grad():
            output = self.model.forward(image)
            probs, index = torch.topk(output, topk)
           
        probs = probs.exp()
        index = index.cpu().numpy()[0]

        output = self.model.forward(image)       
        output = torch.exp(output).data

        probs = torch.topk(output, topk)[0].tolist()[0] # probabilities
        index = torch.topk(output, topk)[1].tolist()[0] # index
        
        idx_to_class = { v : k for k,v in self.model.class_to_idx.items()}
        classes = [idx_to_class[i] for i in index]
        
        return probs, classes
        