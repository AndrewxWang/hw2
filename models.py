# models.py

from sentiment_data import *
from utils import *

from collections import Counter

import numpy as np
import random

from nltk.corpus import stopwords
import nltk

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(stopwords.words('english')) # from nltk
    
    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        feats = Counter() # use counter to encode feature vector

        # takes every word from each sentence (1 by 1 because unigram)
        for word in sentence:
            # normalizes word to loewrcase
            word = word.lower()
            
            # skip empty spaces
            # if word == '':
            #     continue

            # skip stopwords and empty spaces
            # skipping just empty spaces had similar accuracy but double the 
            # run time so i opt to skip stopwords
            if word in self.stopwords or word == '':
                continue

            # if true if we should grow the dimensionality of the featurizer if new features are encountered.
            # adds word to indexer
            if add_to_indexer:
                temp = self.indexer.add_and_get_index(word)
            else:
                temp = self.indexer.index_of(word)
            
            # ignore unseen test words, +1 to seen
            if temp != -1:   
                feats[temp] += 1 # increases counter

        # returns feature vector of words
        return feats


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence, add_to_indexer = False):
        feats = Counter() # use counter to encode feature vector

        # takes every two words adjacent from each sentence (2 because bigram). 
        # cant use for each bc need two words
        for i in range(len(sentence) - 1):
            # normalizes word to lowercase
            word_1 = sentence[i].lower()
            word_2 = sentence[i+1].lower()
            
            # skip stopwords and empty spaces
            # if word_1 in self.stopwords or word_1 == '' or word_2 in self.stopwords or word_2 == '':
            
            # skipping stopwords had less accuracy than just skipping empty spaces
            if word_1 == '' or word_2 == '':
                continue
            
            # create bigram
            bigram = word_1 + " " + word_2

            # if true if we should grow the dimensionality of the featurizer if new features are encountered.
            # adds word to indexer
            if add_to_indexer:
                temp = self.indexer.add_and_get_index(bigram)
            else:
                temp = self.indexer.index_of(bigram)
            
            # ignore unseen test words, +1 to seen
            if temp != -1:   
                feats[temp] += 1 # increases counter

        # returns feature vector of words
        return feats
    
class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.stopwords = set(stopwords.words('english'))

    def get_indexer(self):
        return self.indexer
    
    # Trigram 
    def extract_features(self, sentence, add_to_indexer = False):
        feats = Counter() # use counter to encode feature vector

        # takes every three words adjacent from each sentence (3 because trigram). 
        # cant use for each bc need two words
        for i in range(len(sentence) - 2):
            # normalizes word to lowercase
            word_1 = sentence[i].lower()
            word_2 = sentence[i+1].lower()
            word_3 = sentence[i+2].lower()

            # skip stopwords and empty spaces
            # if word_1 in self.stopwords or word_1 == '' or word_2 in self.stopwords or word_2 == '' or word_3 in self.stopwords or word_3 == '':
            
            # skipping stopwords had less accuracy than just skipping empty spaces
            if word_1 == '' or word_2 == '' or word_2 == '':
                continue
            
            # create bigram
            trigram = word_1 + " " + word_2 + " " + word_3

            # if true if we should grow the dimensionality of the featurizer if new features are encountered.
            # adds word to indexer
            if add_to_indexer:
                temp = self.indexer.add_and_get_index(trigram)
            else:
                temp = self.indexer.index_of(trigram) # else get index of trigram appearance, if no exist, ignore
            
            # ignore unseen test words, +1 to seen
            if temp != -1:   
                feats[temp] += 1 # increases counter
            
        # returns feature vector of words
        return feats


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, bias, featurizer):
        self.weights = weights
        self.bias = bias
        self.featurizer = featurizer

    def predict(self, sentence: List[str]) -> int:
        # perceptron: sign(wx + b)
        feats = self.featurizer.extract_features(sentence, add_to_indexer=False)
        score = self.bias

        for i, val in feats.items():
            if i < len(self.weights):
                score += self.weights[i] * val       

        return 1 if score >= 0 else 0
    
class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, weights, bias, featurizer):
        self.weights = weights
        self.bias = bias
        self.featurizer = featurizer

    def predict(self, sentence: List[str]) -> int:
        # perceptron: sign(wx + b)
        feats = self.featurizer.extract_features(sentence, add_to_indexer=False)
        z = self.bias

        for i, val in feats.items():
            if i < len(self.weights):
                z += self.weights[i] * val       

        # y-hat = 1 if P(y=1|x) > 0.5, otherwise 0
        P = 1 / (1 + np.exp(-z))

        return 1 if P >= 0.5 else 0
    
def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)


    # vocab size
    N  = len(feat_extractor.get_indexer())

    # parameters
    w = np.zeros(N) # weights
    b = 0.0 # initialize bias as 0

    # sign(wx + b) = perceptron

    # # of iterations
    max_iters = 30

    # sets random seed for consistency for testing
    random.seed(10)

    for iter in range(max_iters):
        # randomly shuffles data every epoch
        random.shuffle(train_exs)

        for ex in train_exs:
            # feature vector
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            
            # converts labels to either 1 or -1 since we only care about sign
            y = 1 if ex.label == 1 else -1

            # compute activation
            activation = b

            # sum of d=1 to D of wd + yxd
            for d, x_d in feats.items():
                activation += w[d] * x_d

            # update rule for all d=1 ... D
            if (y * activation) <= 0:
                for d, x_d in feats.items():
                    w[d] += y * x_d
            
                # update bias
                b += 0.008 + y 

                # FOR Q2
                # b += 0.015 + y
                # b += 1/t + y
                # if iter % 4 == 0:
                #     b += 0.02 + y
                # else:
                #     b+= y


    return PerceptronClassifier(w, b, feat_extractor)

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    # vocab size
    N  = len(feat_extractor.get_indexer())

    # parameters
    w = np.zeros(N) # weights
    b = 0.0 # initialize bias as 0

    # sigmoid(wx + b) = LR

    # # of iterations
    max_iters = 30

    # lr for GD
    lr = 0.01

    # sets random seed for consistency for testing
    random.seed(10)

    for iter in range(max_iters):
        # randomly shuffles data every epoch
        random.shuffle(train_exs)

        for ex in train_exs:
            # feature vector
            feats = feat_extractor.extract_features(ex.words, add_to_indexer=False)
            
            # for LR we want a number between 0-1 instead of just sign(wx+b)
            y = ex.label

            # compute wx+b
            z = b

            for d, x_d in feats.items():
                z += w[d] * x_d

            # get sigmoid probability 
            # sigmoid(z) = 1/ (1 + e^-z) gives u a number between 0 and 1

            P = 1 / (1 + np.exp(-z))

            # cross entropy loss used for LR

            # LCE(w,b)= -[y * log sigmoid(wx+b) + (1-y) * log(1- sigmoid(wx+b))]
            error = P - y

            # update weights (w/ Gradient Descent)
            for d, x_d in feats.items():
                w[d] -= lr * error * x_d

            # update bias
            b -= lr * error

    return LogisticRegressionClassifier(w, b, feat_extractor)

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)

        # # Part 1 Q3
        # positive = Beam(10)
        # negative = Beam(10)

        # for i, w in enumerate(model.weights):
        #     word = model.featurizer.get_indexer().get_object(i)
        #     positive.add(word, w)
        #     negative.add(word, -w)
        
        # print("Top 10 positive:", list(positive.get_elts_and_scores()))
        # print("Top 10 negative:", list(negative.get_elts_and_scores()))
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
