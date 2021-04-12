import numpy as np
import os, random, json, time

import torch
from torchvision import datasets, transforms, models
from torch import optim 

from PIL import Image

from Utilities import Utils
#-------------------------------------------------------------------
class NNUtils():
    def __init__(self):
        self.data_dir = None
        self.image_datasets = None 
        self.dataloaders = None
        self.img_resize_to, self.img_crop_to  = 256, 224
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
    #-------------------------------------------------------------------
    @Utils.tryit()
    def cat_to_name(self, cats_filename):
        with open(cats_filename, 'r') as f:
            return json.load(f)
    #-------------------------------------------------------------------
    @Utils.tryit()
    def get_data(self, data_dir):
        self.data_dir = data_dir
        
        train_dir = os.path.join(data_dir, 'train')
        valid_dir = os.path.join(data_dir , 'valid')
        test_dir = os.path.join(data_dir , 'test')

        data_transforms = {'train': transforms.Compose([
                                                transforms.RandomRotation(40),
                                                transforms.RandomResizedCrop(self.img_resize_to), 
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.normalization_mean, self.normalization_std)
                                            ]) , 
                        'valid_or_test': transforms.Compose([
                                                    transforms.Resize(self.img_resize_to),
                                                    transforms.CenterCrop(self.img_crop_to),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(self.normalization_mean, self.normalization_std)])
                         }

        image_datasets = {'train':datasets.ImageFolder(train_dir, transform= data_transforms['train']),
                          'valid': datasets.ImageFolder(valid_dir, transform= data_transforms['valid_or_test']),
                          'test': datasets.ImageFolder(test_dir, transform= data_transforms['valid_or_test'])
                         }

        dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True), 
                       'valid':torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32), 
                       'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
                      }        
        
        self.image_datasets, self.dataloaders = image_datasets, dataloaders
        return self.image_datasets, self.dataloaders
    #-------------------------------------------------------------------
    @Utils.tryit()
    def save_checkpoint(self, save_dir, nnet, image_dataset):   
        nnet.model.class_to_idx = image_dataset.class_to_idx
        checkpoint = {'input_size': nnet.get_classifier_input(), 
                      'output_size': nnet.output_layer.out_features,
                      'hidden_layers': [layer.out_features for layer in nnet.hidden_layers],
                      'arch': nnet.arch,
                      'learning_rate': nnet.learning_rate,
                      'classifier' : nnet.model.classifier,
                      'epochs': nnet.epochs,
                      'optimizer': nnet.optimizer.state_dict(),
                      'state_dict': nnet.model.state_dict(),
                      'class_to_idx': nnet.model.class_to_idx, 
                      'dropout': nnet.dropout_prob,
                     }

        checkpoint_path = os.path.join(save_dir, '{}_checkpoint.pth'.format(nnet.arch))
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    #-------------------------------------------------------------------
    @Utils.tryit()        
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        learning_rate = checkpoint['learning_rate']
        model = getattr(models, checkpoint['arch'])(pretrained=True)
        model.classifier = checkpoint['classifier']
        model.epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        return model, optimizer
    #-------------------------------------------------------------------
    @Utils.tryit()
    def process_image(self, image_path):

        img = Image.open(image_path)
        width, height = img.size

        crop_value = (self.img_resize_to - self.img_crop_to)*0.5 
        img = img.resize((self.img_resize_to, self.img_resize_to))
        img = img.crop((crop_value, crop_value, self.img_resize_to - crop_value, self.img_resize_to - crop_value))

        img = np.array(img)
        img = img/255
        mean = np.array(self.normalization_mean)
        std = np.array(self.normalization_std)
        img = (img - mean) / std
        img = img.transpose((2, 0, 1))
        return img