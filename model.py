# -*- coding:utf-8 -*-
"""
This py file is a skeleton for the main project components.  
Places that you need to add or modify code are marked as TODO
"""

from mmap import ACCESS_DEFAULT
import torch
import torch.nn as nn
import torch.optim
import numpy as np


def word2index(word, vocab):
    """
    Convert a word token to a dictionary index
    """
    if word in vocab:
        value = vocab[word][0]
    else:
        value = -1
    return value


def index2word(index, vocab):
    """
    Convert a dictionary index to a word token
    """
    for w, v in vocab.items():
        if v[0] == index:
            return w
    return 0



class Model(object):
    def __init__(self, args, vocab, trainlabels, trainsentences, testlabels, testsentences):
        """ The Text Classification model constructor """
        self.embeddings_dict = {}
        self.datarep = args.datarep
        if self.datarep == "GLOVE":
            print("Now we are using the GloVe embeddings")
            self.load_glove(args.embed_file)
        else:
            print("Now we are using the BOW representation")
        self.vocab = vocab
        self.trainlabels = trainlabels
        self.trainsentences = trainsentences
        self.testlabels = testlabels
        self.testsentences = testsentences
        self.lr = args.lr
        self.embed_size = args.embed_size
        self.hidden_size = args.hidden_size
        self.traindataset = []
        self.testdataset = []

        """
        TODO
        You should modify the code for the baseline classifiers for self.datarep
        shown below, which is a three layer model with an input layer, a hidden layer,
        and an output layer. You will need at least to define the dimensions for
        the size of the input layer (ISIZE; see where this is passed in by argparse),
        and the hidden layer (e.g., HSIZE).  Do not change the size of the output
        layer, which is currently 2, as this corresponds to the number of sentiment classes.
        You need to choose an activation function. Once you get this working
        by uncommenting these lines, adding the activation function, and replacing
        ISIZE and HSIZE, see if you can achieve the classification accuracy on movies
        of 0.85 for the GLoVe representation, or 0.90 for the BOW representation.  
        You are free to modify the code for self.model, e.g., to add more hidden layers, or 
        to change the input representation created in prepare_datasets(), to raise the accuracy.
        """

        ISIZE = self.embed_size
        HSIZE = self.hidden_size

        if self.datarep == "GLOVE":
            self.model = nn.Sequential(
            nn.Linear(ISIZE, HSIZE),
            # TODO insert a line for the activation function; you will need to look
            # at the pytorch documentation
            nn.ReLU(),
            nn.Linear(HSIZE, 2),
            nn.LogSoftmax(dim=1))
        else:
            self.model = nn.Sequential(
            nn.Linear(ISIZE, HSIZE),
            # TODO insert a line for the activation function; you will need to look
            # at the pytorch documentation
            nn.ReLU(),
            nn.Linear(HSIZE, 2),
            nn.LogSoftmax(dim=1))
        

    def prepare_datasets(self):
        """
        Load both training and test
        Convert the text spans to BOW or GLOVE vectors
        """

        datasetcount = 0

        for setOfsentences in [self.trainsentences, self.testsentences]:

            sentcount = 0
            datasetcount += 1

            for sentence in setOfsentences:
                sentcount += 1
                # vsentence holds lexical (GLOVE) or word index (BOW) input to sentence2vec
                vsentence = []
                for l in sentence:
                    if l in self.vocab:
                        if self.datarep == "GLOVE":
                            vsentence.append(l)
                        else:
                            vsentence.append(word2index(l, self.vocab))
                svector = self.sentence2vec(vsentence, self.vocab)
                if (len(vsentence) > 0) & (datasetcount == 1): # train
                    self.traindataset.append(svector)
                elif (len(vsentence) > 0) & (datasetcount == 2): # test
                    self.testdataset.append(svector)

        print("\nDataset size for train: ",len(self.traindataset)," out of ",len(self.trainsentences))
        print("\nDataset size for test: ",len(self.testdataset)," out of ",len(self.testsentences))
        indices = np.random.permutation(len(self.traindataset))

        self.traindataset = [self.traindataset[i] for i in indices]
        self.trainlabels = [self.trainlabels[i] for i in indices]
        self.trainsentences = [self.trainsentences[i] for i in indices]

    def rightness(self, predictions, labels):
        """ 
        Error rate
        """
        pred = torch.max(predictions.data, 1)[1]
        rights = pred.eq(labels.data.view_as(pred)).sum()
        return rights, len(labels)

    def sentence2vec(self, sentence, dictionary):
        """ 
        Convert sentence text or indices to vector representation
        """
        """
        #TODO 
        You should modify the code to define two methods to convert the review text to a vector:
        one is for Glove and another is for BOW. The first step is to set the size of the vectors, 
        which will be different for GLOVE and BOW. The next step is to create the vectors for your input sentences.
        Hint: Use numpy to init the vector. Retrieve the BOW vector from self.vocab defined as part of the init for the 
        class, and write a function to create the vector values. Retrieve the GLOVE word vectors from the 
        embeddings_dict created by the load_glove(path) function

        # Code:
        sentence: List of words (tokens) in the sentence.
        dictionary: Dictionary representing the vocabulary mapping words to indices.
        """

        if self.datarep == "GLOVE":
            #TODO
            # Retrieve the GLOVE word vectors from the embeddings_dict
            vectors = []
            for token in sentence:
                if token in self.embeddings_dict:
                    vectors.append(self.embeddings_dict[token])
            # Calculate the mean of the vectors to create the final sentence vector
            if vectors:
                vector = np.mean(vectors, axis=0)
            else:
                # If no words in the sentence are in the embeddings_dict, return a zero vector
                vector = np.zeros(self.embed_size)
            return vector

        else:
            #TODO if self.datarep == "BOW"
            # Initialize a zero vector using numpy with size equal to the vocabulary size
            vector = np.zeros(len(dictionary), dtype=int)
            # Increment the position in the vector for each word in the sentence
            for token in sentence:
                if token in dictionary:
                    vector[dictionary[token]] += 1
            # return: BOW representation of the sentence as a numpy array
            return vector

    def load_glove(self, path):
        """
        Load Glove embeddings dictionary
        """
        """
        You should load the Glove embeddings from the local glove files - eg "glove.6B.100d", 
        Use "self.embeddings_dict" to store the glove vector dictionary.
        """
        with open(path, 'r') as f:
            # TODO
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self.embeddings_dict[word] = vector
        return 0

    def training(self):
        """
        The training and testing process.
        """
        losses = []
        """
        # TODO
        loss_function = LOSS
        optimizer = OPTIMIZER

        Note that the learning rate (lr) for the optimizer is a command line parameter.

        Decide on the number of training epochs; it can be the same for both representations, or different
        # TODO
        if self.datarep == "GLOVE":
            tr_epochs = TR
        else:
            tr_epochs = TR
        """
        loss_function = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        if self.datarep == "GLOVE":
            tr_epochs = 10
        else:
            tr_epochs = 10

        for epoch in range(tr_epochs):
            print(epoch)
            for i, data in enumerate(zip(self.traindataset, self.trainlabels)):
                x, y = data
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                optimizer.zero_grad()
                # predict
                predict = self.model(x)
                # calculate loss
                loss = loss_function(predict, y)
                losses.append(loss.data.numpy())
                loss.backward()
                optimizer.step()
                # test every 1000 data
                if i % 1000 == 0:
                    test_losses = []
                    rights = []
                    for j, test in enumerate(zip(self.testdataset, self.testlabels)):
                        x, y = test
                        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                        y = torch.tensor(np.array([y]), dtype=torch.long)
                        predict = self.model(x)
                        right = self.rightness(predict, y)
                        rights.append(right)
                        loss = loss_function(predict, y)
                        test_losses.append(loss.data.numpy())

                    right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                    print('At epoch {}: Training loss：{:.2f}, Test loss：{:.2f}, Test Acc: {:.2f}'.format(epoch, np.mean(losses),
                                                                                np.mean(test_losses), right_ratio))
        print("End Testing/Training")




