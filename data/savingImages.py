import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import pandas as pd
import skimage.io as io
from tqdm import tqdm



#number of images we are going to create in each of the two classes
nfigs = 5000
ntest = 1000

# square image dimension
size=256

lengths = np.array(pd.read_csv('length_data_87k_examples_256px.csv'))
urlsTrain = [x.replace("{}/ml/length/", "") for x in lengths[:5000,4]]
labelsTrain = lengths[:5000,1]

urlsTrainSeg = dict()
for c in range(7):
    urlsTrainSeg[c] = []

for k in range(len(labelsTrain)):
    urlsTrainSeg[labelsTrain[k]].append(urlsTrain[k])

urlsTest = [x.replace("{}/ml/length/", "") for x in lengths[-1000:,4]]
labelsTest = lengths[-1000:,1]

urlsTestSeg = dict()
for c in range(7):
    urlsTestSeg[c] = []

for k in range(len(labelsTest)):
    urlsTestSeg[labelsTest[k]].append(urlsTrain[k])

#loop over classes
for c in range(7):
    clss = str(c)
    print("generating images of "+clss+":")
    #loop over number of images in the class
    for i in tqdm(range(len(urlsTrainSeg[c]))):
        img = io.imread(urlsTrainSeg[c][i])
        io.imsave('data/train/'+clss+'/data'+str(i)+'.jpeg',img, quality=100)
    for i in tqdm(range(len(urlsTestSeg[c]))):
        img = io.imread(urlsTestSeg[c][i])
        io.imsave('data/test/'+clss+'/data'+str(i)+'.jpeg',img, quality=100)