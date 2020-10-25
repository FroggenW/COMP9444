#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating
additional variables, functions, classes, etc., so long as your code
runs with the hw2main.py file unmodified, and you are only using the
approved packages.

You have been given some default values for the variables stopWords,
wordVectors(dim), trainValSplit, batchSize, epochs, and optimiser.
You are encouraged to modify these to improve the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe

import torch

from nltk.stem.porter import *
stemmer = PorterStemmer()
import re

###########################################################################
### The following determines the processing of input data (review text) ###
###########################################################################

# Authors: Chang Wang, Yan Zhao

# First of all, in the preprocess part, we process the review,
# such as removing stopwords, filtering stems, such as in ing format, and removing words less than 3 in length,
# because these words are basically redundant.
# Secondly, for the postprocess part, we pad each word to achieve a feature enhancement effect

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # remove all the punctuations and numbers, only keep the words
    # sample = [stemmer.stem(i) for i in sample]
    # sample = [w for w in sample if not w in stopWords]
    # print('123', sample)
    out = []
    for i in range(len(sample)):
        word = re.sub(r'[^a-z]', "", sample[i])
        if word != "":
            out.append(word)
    print(out)
    return out

def postprocessing(batch, vocab):
    """
    Called after numericalisation but before vectorisation.
    """
    # max_len = max([len(a) for a in batch])
    # l = []
    # for a in batch:
    #     res = max_len - len(a)
    #     if res > 0:
    #         a.extend([['a'] * len(a[0])] * res)
    #     l.append(a)
    # l = [sentence for sentence in l if len(sentence) > 0]
    # return l

    return batch

stopWords = {}
#stopWords = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u'should', u'now']
wordVectors = GloVe(name='6B', dim=100)

###########################################################################
##### The following determines the processing of label data (ratings) #####
###########################################################################

def convertLabel(datasetLabel):
    """
    Labels (product ratings) from the dataset are provided to you as
    floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    You may wish to train with these as they are, or you you may wish
    to convert them to another representation in this function.
    Consider regression vs classification.
    """

    return datasetLabel.round()

def convertNetOutput(netOutput):
    """
    Your model will be assessed on the predictions it makes, which
    must be in the same format as the dataset labels.  The predictions
    must be floats, taking the values 1.0, 2.0, 3.0, 4.0, or 5.0.
    If your network outputs a different representation or any float
    values other than the five mentioned, convert the output here.
    """

    return netOutput.round()

###########################################################################
################### The following determines the model ####################
###########################################################################



class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network
    will be a batch of reviews (in word vector form).  As reviews will
    have different numbers of words in them, padding has been added to the
    end of the reviews so we can form a batch of reviews of equal length.
    """

    def __init__(self):
        super(network, self).__init__()
        # self.lstm = tnn.LSTM(
        #     input_size=300,
        #     hidden_size=100,
        #     batch_first=True,
        # )
        # self.fc1 = tnn.Linear(100, 50)
        # self.fc3 = tnn.Linear(50, 1)
        # self.tanh = tnn.Tanh()
        self.lstm = tnn.LSTM(input_size=100, hidden_size=100, batch_first=True)
        self.linear = tnn.Linear(100,1)

    def forward(self ,input,length):
        # out, (hn, cn) = self.lstm(input)
        # output = self.tanh(self.fc1(hn[-1]))
        # output = self.fc3(output)
        
        # lstm layer
        outputs,(h_n,c_n) = self.lstm(input)
        # linear layer
        output = self.linear(h_n[-1])
        return output.squeeze()

#
# class loss(tnn.Module):
#     """
#     Class for creating a custom loss function, if desired.
#     You may remove/comment out this class if you are not using it.
#     """
#
#     def __init__(self):
#         super(loss, self).__init__()
#
#     def forward(self, output, target):
#         pass
# we selected L1Loss function

def loss():
    return tnn.MSELoss()
    #return tnn.CrossEntropyLoss()

net = network()
"""
    Loss function for the model. You may use loss functions found in
    the torch package, or create your own with the loss class above.
"""

lossFunc = loss()

###########################################################################
################ The following determines training options ################
###########################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.Adam(net.parameters(), lr=0.0016)

