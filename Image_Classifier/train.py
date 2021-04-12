import os, random, argparse, time

from NNUtilities import NNUtils
from NNetwork import NNetwork
from Utilities import Utils
#-------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type = str, help='path to images directory')
parser.add_argument('--save_dir', type = str, help='path to directory where (.pth) checkpoints files will be saved' )
parser.add_argument('--arch', type = str, default='vgg16', choices=['vgg16', 'alexnet', 'densenet161'], help="model name")
parser.add_argument('--gpu', action='store_true', default=False, help='use GPU during model training')
parser.add_argument('--epochs', type = int, default=8,  help='number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_units', type=int, nargs='+', default=1024, help='Number of nodes for each hidden layer')
parser.add_argument('--output_size', type=int, default=102, help='Number of output nodes (Flower Classes/Labels)')
args=parser.parse_args()
#-------------------------------------------------------------------
@Utils.tryit()
def train():
    nnutils = NNUtils()
    image_datasets, dataloaders = nnutils.get_data(args.data_dir)

    nnet = NNetwork(args.gpu)
    nnet.build_model(args.arch, hidden_units=args.hidden_units, classifier_output=args.output_size, learning_rate=args.learning_rate)
    Utils.log(0, 'model created')
    for c in nnet.model.classifier.children():
        Utils.log(1, str(c))

    Utils.log(0, "Training will start using {}", "GPU" if(nnet.gpu) else "CPU")
    nnet.train_model(args.epochs, dataloaders['train'], dataloaders['valid'], args.gpu)    
    nnet.test_model_accuracy(dataloaders['test'])
    checkpoint_path = nnutils.save_checkpoint(args.save_dir, nnet, image_datasets['train'])
    Utils.log(0, 'checkpoint saved to file "{}"', checkpoint_path)
#-------------------------------------------------------------------
if __name__ == '__main__': 
    train()
