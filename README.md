# Semi-Supervised Learning : Self-Training
Self Training and propagating one nearest neighbor
(강필성 교수님 Business Anlaytics 수업 일환)

### Requirements
python == 3.6
tensorflow == 1.12.0
numpy >= 1.15
matplotlib


### self_training.py 
: data - MNIST data from tensorflow 
method - change method to compare different variations
a) all : use all the unlabeled data and predicted label
b) top : use top 10% (1000) well-classified data and their label
c) weights : use all the unlabeled data and predicted logit as label

### onenn.py
: data - randomly created distribution
method - propagating one nearest neighbor
out - plot before & after
