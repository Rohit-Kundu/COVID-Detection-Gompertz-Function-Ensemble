from utils_ensemble import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default = './', help='Directory where csv files are stored')
parser.add_argument('--topk', type=int, default = 2, help='Top-k number of classes')
args = parser.parse_args()

root = args.data_directory
if not root[-1]=='/':
    root=root+'/'

p1,labels = getfile(root+"vgg11")
p2,_ = getfile(root+"wideresnet50-2")
p3,_ = getfile(root+"inception")

#Check utils_ensemble.py to see the "labels" distribution. Change according to the dataset used. By default it has been set for the SARS-COV-2 dataset.

#Calculate Gompertz Function Ensemble
top = args.topk #top 'k' classes
predictions = Gompertz(top, p1, p2, p3)

correct = np.where(predictions == labels)[0].shape[0]
total = labels.shape[0]

print("Accuracy = ",correct/total)
classes = []
for i in range(p1.shape[1]):
    classes.append(str(i+1))
metrics(labels,predictions,classes)

plot_roc(labels,predictions)
