import os, random, argparse

from NNUtilities import NNUtils
from NNetwork import NNetwork
from Utilities import Utils
#-------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('input', action='store', help='path to image file to be classified')
parser.add_argument('checkpoint', action='store', help='path to (.pth) checkpoint file to load model from')
parser.add_argument('--topk', action='store', type=int, default=5, help='top most probabilities and classes')
parser.add_argument('--category_names', action='store', default="cat_to_name.json", help='JSON file with name of each class')
parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for prediction')
args=parser.parse_args()

#-------------------------------------------------------------------
@Utils.tryit()
def predict():
    nnutils = NNUtils()
    nnet = NNetwork(args.gpu)

    cat_to_name = nnutils.cat_to_name(args.category_names)
    nnet.cat_to_name = cat_to_name

    Utils.log(0, "Predicting image using {}", "GPU" if(nnet.gpu) else "CPU")
    nnet.model, nnet.optimizer = nnutils.load_checkpoint(args.checkpoint)
    Utils.log(0, 'model loaded from checkpoint "{}"', args.checkpoint)
    for c in nnet.model.classifier.children():
        Utils.log(1, str(c))

    
    #_dir = "./flowers/test/{}/".format(random.randint(1,102))
    #args.input = "{}{}".format(_dir, random.choice(os.listdir(_dir)))

    img = nnutils.process_image(args.input)

    probs, classes = nnet.predict(img, args.topk, args.gpu)
    class_names = [cat_to_name[clss] for clss in classes]

    Utils.log(0, 'topk {} of image: "{}"',args.topk, args.input)
    Utils.log(1, "Probabilities: {}", ["{:.5f}".format(p) for p in probs])
    Utils.log(1, "Classes: {}", classes)
    Utils.log(1, "Categories: {}", class_names)

    max_idx = probs.index(max(probs))
    Utils.log(0, 'Flower Class: "{}" - Category: "{}" - Probability: "{}"', classes[max_idx], class_names[max_idx], probs[max_idx])
#-------------------------------------------------------------------

if __name__ == '__main__': 
    predict()