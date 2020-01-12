# -*- coding: utf-8 -*-
"""
Part 1, AI Project 1
"""
# importing the libraries I need
import numpy as np
import matplotlib.pyplot as plt
plt.rc('figure', figsize=(14, 8))
plt.rc('grid', color='gray', alpha = 0.3, linestyle='solid')
import os
import pandas as pd
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import word_tokenize

# Setting working directory
root_dir = '/home/luca/Projects/Project 1 AI/Part 3'
os.chdir(root_dir)


##########################################################
# DATA
##########################################################


# Defining a file reader to read train, dev and test data
def read_file(file_name):
    '''
    Takes as input either the train, dev or test file names and outputs a list of lists
    containg datapoints consisting of a sentence (string) and a target (an integer)
    '''
    with open(file_name,'r', encoding = "utf-8") as text_file:
        l = []
        for line in text_file:
            l.append(line.rstrip().split('\t'))
            
        # Transforming target variables from stings to integers
        l = [[sentence, int(target)] for sentence, target in l]
        text_file.close()
        
        return(l)

# loading train, dev and test data
train_list = read_file('senti_binary.train')
dev_list = read_file('senti_binary.dev')
test_list = read_file('senti_binary.test')

# To deal with apostrophes after tokenization has taken place

apostrophe_dict = {
"n't" : "not",
"'d"  : "would",
"'ll" : "will",
"'m"  : "am",
"'s"  : "is",
"'re" : "are",
"'ve" : "have",
"wo"  : "will",  # won't -> [wo, n't] -> [will, not]
"sha" : "shall"  # shan't -> [sha, n't] -> [shall, not]
}

# Defining a useful function to preprocess the text of the sentences
def text_preprocess(sentence):
    
    #1.st step: lowercase
    sentence_lower = sentence.lower()
    
    #2.nd sep: replace '-' with ' ' so we won't lose words in step 5
    sentence_replace = sentence_lower.replace('-',' ')
    
    # 3.rd step: tokenization
    sentence_token = word_tokenize(sentence_replace)
    
    # 4.th step: apostrophe handling
    setence_apostrophe = [apostrophe_dict[word] if word in apostrophe_dict else word for word in sentence_token]
    
#########    
    # I will no remove stop words because I tried and keeping them improves the accuracy of the models
    
    # 5.th step: removing stop words
#     sentence_stop = [word for word in setence_apostrophe if word not in stopwords.words('english')]
#########    
    
    # 6.th step: removing non alphabetic characters
    sentence_alpha = [word for word in setence_apostrophe if word.isalpha()]
    
    return(sentence_alpha)
    
# Let's now build our dictionary of words.

# First, preprocess all the sentences in the train set
train_sentences_list = [text_preprocess(datapoint[0]) for datapoint in train_list]

# Getting the length of the sentences (it will be useful later)
train_sentences_lengths = np.array([len(sentence) for sentence in train_sentences_list])

# Now, flatten the list, remove the duplicates and sort the list
train_words_list = [word  for sentence in train_sentences_list for word in sentence]

train_unique_words_list = []
for word in train_words_list:
    if not (word in train_unique_words_list):
        train_unique_words_list.append(word)

train_unique_words_list.sort()

# Finally, let's create the dictionary which associates
word_to_idx = {word: (i+1) for i, word in enumerate(train_unique_words_list)}

# Let's add to the thictionary, to index 0, a word called __unknown__ for which we will map all the words we don't know
word_to_idx['__unknown__'] = 0

# Let's define a dictionary to retrieve the words from the indexes
idx_to_word = {word_to_idx[word]: word for word in word_to_idx} 

# Print the length of the dictionary:
print('The length of our dictionary is: ', len(word_to_idx))

plt.hist(train_sentences_lengths);

print('Max length of a sentence is: ', np.max(train_sentences_lengths))

# Defining some useful functions

def tokenized_sentence_to_idx_tensor(tokenized_sentence):
    '''
    Takes in a list of strings and returns a 50-dimensional long tensor of indexes (of our dictionary).
    If the word are unknown or the sentence is shorter than 50, the indexes returned will be that of the
    unknown word. In our case 0.
    '''
    idx_list = []
    for i in range(50):
        
        try:
            word = tokenized_sentence[i]
            idx = word_to_idx[word]
            idx_list.append(idx)
        except:
            idx_list.append(0)
    
    return torch.LongTensor(idx_list)


def sentence_to_idx_tensor(sentence):
    '''
    Takes in a sentence in form of a string, preprocesses it and then  returns a 50-dimensional 
    long tensor of indexes (of our dictionary). If the word are unknown or the sentence is shorter than 50,
    the indexes returned will be that of the unknown word. In our case 0.
    '''
    return(tokenized_sentence_to_idx_tensor(text_preprocess(sentence)))
    
# Let's build a dataset class (Inherits from torchs' Dataset class)

class review_dataset(Dataset):
    
    def __init__(self, data, transform_features = None, transform_labels = None, n_words_in_sentence = 50):
        super().__init__()
        self.data = data
        self.n_words_in_sentence = n_words_in_sentence
        self.transform_features = transform_features
        self.transform_labels = transform_labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        features = self.data[index][0]
        labels = self.data[index][1]
        
        if self.transform_features:
            features = self.transform_features(features)
                
        if self.transform_labels:
            labels = self.transform_labels(labels)
        
        return (features, labels)
    
    def get_tensors_dataset(self, numpy=False):
        '''
        Returns the actual dataset made up by tensors that is used internally by the NN (useful when we want to move computations on gpu)
        It il a list made up by 2 elements:
        1.st element is tensor of features
        2.nd element is the tensor of labels
        
        if numpy==True  numpy arrays instead of torch tensors are returned (Useful if we want to fit sklearn models)
        '''
        dataset_list = [self.__getitem__(i) for i in range(len(self))]
        features_list = [datapoint[0].reshape(1,-1) for datapoint in dataset_list]
        classes_list = [datapoint[1] for datapoint in dataset_list]
        
        if numpy==False:
            return (torch.cat(features_list), torch.LongTensor(classes_list))

        elif numpy==True:
            return (torch.cat(features_list).numpy(), torch.LongTensor(classes_list).numpy())
            
            
# Define instances of my reviews_dataset class
train = review_dataset(train_list, sentence_to_idx_tensor)
dev = review_dataset(dev_list, sentence_to_idx_tensor)
test = review_dataset(test_list, sentence_to_idx_tensor)

# Define a dataloader to make the training in batches

batch_size=64

train_loader = DataLoader(batch_size=batch_size,
                          dataset=train,
                          shuffle=True,
                          drop_last=True
                         )

#####################################################################
# MODEL (Superclass)
#####################################################################

# Let's create a new superclass that inherits from nn.Module. 
# We will add the fit, generate_pictures, predict_prob and predict methods to it, to reduce lines of code

class neural_network_sentiment(nn.Module):
    def __init__(self):
        super().__init__()
    
    def predict_prob(self, review):
        '''
        Takes in a string (theoretically a film review) and outputs a tensor representing 
        the probability distribution over 0 (negative review) and 1 (positive review)
        '''
        prob_distribution = F.softmax(self.forward(sentence_to_idx_tensor(review)), dim=0)
        return prob_distribution.data
    
    def predict(self, review):
        '''
        Takes in a string (theoretically a film review) and outputs an integer representing 
        the predicted class: 0 (negative review) or 1 (positive review)
        '''
        return int(torch.max(self.forward(sentence_to_idx_tensor(review)), 0)[1].data.numpy())
    
    def fit(self,
            train_loader,
            test_loader,
            dev_loader=None,
            initial_lr = 1,
            gamma = 0.95,
            n_epochs=30,
            score = False,
            gpu=False,
            print_results = True,
            chart=True):
        
        '''
        Takes in a dataloader for the training set, one for the test, set (if gpu==True they
        should be already on the gpu) and trains the network.
        score = True -> self.score will be defined as the accuracy on the dev set (at the last epoch).
        When score = True, you should plug in a dev_loader, or the score will be computed on the test
        print_results = True -> shows how the training is going every epoch
        chart=True -> shows a chart (1x2) representing epoch vs train and test loss and accuracy
        '''
        
        if gpu==True:
            # Move the model to gpu (the datasets are supposed to be already been moved on the gpu)
            self.to(device)            
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=initial_lr, momentum=0.1)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = gamma)
        
        self.train_loss = [0 for i in range(n_epochs)]
        self.test_loss = []
        self.train_accuracy = []
        self.test_accuracy = []
        
        self.n_epochs = n_epochs
        self.initial_lr = initial_lr
        self.gamma = gamma
        
        for epoch in range(n_epochs):
                
            train_correct = 0
            train_total = 0
            
            for i, (items, classes) in enumerate(train_loader):
                                    
                self.train()                             # begin training
                     
                self.optimizer.zero_grad()                # zero out gradients
                outputs = self(items)                     # forward pass
                loss = self.criterion(outputs, classes)   # get loss
                loss.backward()                           # update gradients (backprop)
                self.optimizer.step()                     # update parameters based on gradients
        
                # computes accuracy on training
                train_total += classes.size(0)            # counts the number of rows we trained
                _, predicted = torch.max(outputs.data, 1) # stores the predicted classes
                train_correct += (predicted == classes.data).sum()  # counts how many times we got it right
        
                # adds up training losses for the batches (divided by the number of batches in order to get the mean loss of the epoch)
                self.train_loss[epoch] += loss/len(train_loader)
                

            self.eval()                                   # begin evaluation
    
            # save accuracy on training
            self.train_accuracy.append(100 * (train_correct.cpu().item() / train_total))
    
            # test set predictions
            outputs = self(test_loader[0])
    
            # compute and save loss on test
            loss = self.criterion(outputs, test_loader[1])
            self.test_loss.append(loss.item())
    
            # compute and save accuracy on test
            _, predicted = torch.max(outputs.data, 1)
            total = test_loader[1].size(0)
            correct = (predicted == test_loader[1]).sum()
            self.test_accuracy.append((100 * correct.cpu().item() / total))
            
            if print_results == True:
                if ((epoch+1)%1 == 0) or epoch==0:
                    print ('Epoch %d/%d, Accuracy on Training Set: %.4f, Accuracy on Test Set: %.4f'
                          %(epoch+1, n_epochs, self.train_accuracy[epoch], self.test_accuracy[epoch]))                    
              
            # Update learning rate
            self.scheduler.step()
            
            # Free memory
            if gpu == True:
                torch.cuda.empty_cache()
            gc.collect()   
       
        
        if chart == True:
            print('\n')
            
            fig, ax = plt.subplots(1,2)
            
            epochs = range(1,self.n_epochs +1)

            # train and test loss
            ax[0].plot(epochs, self.train_loss, label='train loss', c = 'red')
            ax[0].plot(epochs, self.test_loss, label='test loss', c = 'blue')
            ax[0].set_title("Train and Test Loss")
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Loss')
            ax[0].legend()

            # train and test accuracy
            ax[1].plot(epochs, self.train_accuracy, label='train accuracy', c = 'red')
            ax[1].plot(epochs, self.test_accuracy, label='test accuracy', c = 'blue')
            ax[1].set_title("Train and Test Accuracy")
            ax[0].set_xlabel('Epoch')
            ax[0].set_ylabel('Accuracy')
            ax[1].legend(); 
            
            self.chart = fig
        
        if score == True:
            if dev_loader:
                outputs = self(dev_loader[0])
                _, predicted = torch.max(outputs.data, 1)
                total = dev_loader[1].size(0)
                correct = (predicted == dev_loader[1]).sum()
                self.score = 100 * (correct.cpu().item() / total)
            else:
                self.score = self.test_accuracy[-1]
                
        # Move again the neural network on the cpu (to make the other methods work)
        if gpu == True:
            torch.cuda.empty_cache() 
            self.cpu()
            try:
                self.attention_weights.cpu()
            except:
                pass
            
    def generate_training_pictures(self):
                
        '''
        This methon can oly be used after the fit method has been called. It generates two pictures
        that show the dynamics of the loss and accuracy for train and test set during training
        '''
        
        plt.rc('figure', figsize=(14, 8))
        plt.rc('grid', color='gray', alpha = 0.3, linestyle='solid')
            
        fig1, ax1 = plt.subplots()
            
        epochs = range(1,self.n_epochs +1)
            
        ax1.plot(epochs, self.train_loss, label='train loss', c = 'red')
        ax1.plot(epochs, self.test_loss, label='test loss', c = 'blue')
        ax1.set_title("Train and Test Loss")
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
            
        fig2, ax2 = plt.subplots()
            
        ax2.plot(epochs, self.train_accuracy, label='train accuracy', c = 'red')
        ax2.plot(epochs, self.test_accuracy, label='test accuracy', c = 'blue')
        ax2.set_title("Train and Test Accuracy")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend(); 
            
        self.fig1 = fig1
        self.fig2 = fig2
        
######################################################
# TRAINING (Chose from cpu and gpu)
######################################################
        
# Set cpu = True or False whether you want to use cpu or not
        
##################
cpu = True
##################

# Let's move the datasets to the GPU in case we want to train the model there

# Initializes the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

gpu = not cpu
if cpu == True:
    device = "cpu"

# Creating tensors for the dev and test set
dev_loader = dev.get_tensors_dataset()
test_loader = test.get_tensors_dataset()

if gpu == True:
    # Move the training set dataloader to the gpu
    train_loader = [(features.to(device), labels.to(device)) for (features,labels) in train_loader]

    # Moving tensors for the dev set
    dev_loader = [element.to(device) for element in dev_loader]

    # Moving tensors for the test set
    test_loader = [element.to(device) for element in test.get_tensors_dataset()]

# Now let's try to build a very simple model that learns the "sentiment-based word embeddings", weights the word embeddings by their importance in the sentence, and then output a sentiment for the sentence 
# Averaging these embeddings and then input them in a simple perceptron classifier

class neural_network(neural_network_sentiment):
    
    def __init__(self, dictionary_length, n_embeddings, n_classes):
        super().__init__()
        self.dictionary_length = dictionary_length
        self.n_embeddings = n_embeddings
        
        self.attention_vector = nn.Parameter(data=torch.Tensor(n_embeddings).float().to(device), requires_grad=True)
        self.attention_vector.data.uniform_(-1, 1) # uniformly initialize the attention vector
        
        self.n_classes = n_classes
        
        # padding_idx=0 makes sure the embedding for the __unknown__ word is 0
        self.embeddings = nn.Embedding(dictionary_length, self.n_embeddings, padding_idx=0, sparse=True) 
        self.fc1 = nn.Linear(self.n_embeddings, self.n_classes)

    def attention_layer(self, embeds):
        if len(embeds.shape) == 2:    # If we feed the network a single example
            
            # First step: compute score
            score = []
            for word_embedding in embeds:
                score.append(torch.dot(word_embedding, self.attention_vector))
                
            weights_non_normalized = torch.Tensor(score)
            
            # Second step: compute weights
            weights_normalized = F.softmax(weights_non_normalized, dim=0).reshape(-1,1)
            
            # Third step, compute weighted average
            mean_embeds = (weights_normalized * embeds).sum(dim=0).reshape(-1)
            
        elif len(embeds.shape) == 3:   # If we feed the network a batch of examples
            
            # Retrieve the dimension of a Batch
            batch_size = embeds.shape[0]
            
            mean_embeds = []
            for i in range(batch_size):
                
                # First step: compute score
                score = []
                for word_embedding in embeds[i]:
                    score.append(torch.dot(word_embedding, self.attention_vector))
                weights_non_normalized = torch.Tensor(score)
                # Second step: compute weights
                weights_normalized = F.softmax(weights_non_normalized, dim=0).reshape(-1,1)
                # Third step, compute weighted average
                me = (weights_normalized * embeds[i]).sum(dim=0).reshape(1,-1)
                mean_embeds.append(me)
                
            mean_embeds = torch.cat(mean_embeds, 0)
                
        return mean_embeds
    
    
    def forward(self, x):
        
        # Embedding layer
        embeds = self.embeddings(x)
        
        # Weighting Layer
        mean_embeds = self.attention_layer(embeds)
        
        # Linear layer
        out = self.fc1(mean_embeds)
        
        return(out)
        
# Instantiate the net
my_neural_network = neural_network(len(word_to_idx), 10, 2)

# Train the net (uncomment)
my_neural_network.fit(train_loader, test_loader, gpu=gpu, initial_lr = 10, gamma = 0.85)
torch.save(my_neural_network.state_dict(), "./part3_state.chkpt")

#############################################################
# Compute attention weights and spot trends
#############################################################

# Reload the model (do not run this if you haven't run the script yet)
#my_neural_network.load_state_dict(torch.load(os.getcwd() + '/to_submit/part3_state.chkpt'))

def get_attention_weights(sentence):
    '''
    Takes in a sentence (string) returns the weights assigned to each word 
    according to the model we trained before (my_embedings_perceptron).
    Note that we discard duplicates so that weights won't sum to one
    '''           
    
    word = text_preprocess(sentence)
    idx_tensor = tokenized_sentence_to_idx_tensor(word)
    length_sentence = len(word) + 1
    embeds = my_neural_network.embeddings(idx_tensor)
    
    score = []
    for word_embedding in embeds:
        score.append(torch.dot(word_embedding, my_neural_network.attention_vector))
    weights_non_normalized = torch.Tensor(score)
    weights_normalized = F.softmax(weights_non_normalized, dim=0).reshape(-1,1)
    
    tensor = torch.cat([idx_tensor.float().reshape(-1,1), weights_normalized.reshape(-1,1)], dim = 1)[:length_sentence]

    return tensor

# Now let's compute the weights for the sentences in the dev data

weights_dev_tensor = torch.cat([get_attention_weights(sentence) for sentence, target in dev_list], dim=0)
weights_dev_df = pd.DataFrame(weights_dev_tensor.numpy(), columns=['idx','weight'])

weight_distributions_dict = {}
for word_idx in idx_to_word:
    dist = weights_dev_df[weights_dev_df.idx == word_idx]['weight'].values
    weight_distributions_dict[word_idx] = dist

# Remove words that that appeared less than 5 times
weight_distributions_dict = {word_idx: weight_distributions_dict[word_idx] for word_idx in weight_distributions_dict if not weight_distributions_dict[word_idx].size < 5}

# Compute mean and sd of the weights distributions
weight_means_sd_dict = {word: [weight_distributions_dict[word].mean(),weight_distributions_dict[word].std()] for word in weight_distributions_dict}

# Create a dataframe to analize the data
weight_mean_sd_df = pd.DataFrame.from_dict(weight_means_sd_dict, orient='index', columns=['mean','std'])
# Introduce a cv column (the coefficient of variation (std/abs(mean)) (abs isn't needed because all weights are positive))
weight_mean_sd_df['cv'] = weight_mean_sd_df['std']/weight_mean_sd_df['mean']
# Introduce a word column
weight_mean_sd_df['word'] = list(map(lambda x: idx_to_word[x], weight_mean_sd_df.index))

print('Number of words we are analyzing: ', len(weight_mean_sd_df), '\n')

# Sort by standart deviation
weight_mean_sd_df = weight_mean_sd_df.sort_values(by = ['std'])

print('\nlow std weights words: \n')
print(weight_mean_sd_df.head(10))
print('\nhigh std weights words: \n')
print(weight_mean_sd_df.tail(10))

# Now sort by mean
weight_mean_sd_df = weight_mean_sd_df.sort_values(by = ['mean'])

print('\nlow mean weights words: \n')
print(weight_mean_sd_df.head(10))
print('\nhigh mean weights words: \n')
print(weight_mean_sd_df.tail(10))

# Now sort by cv
weight_mean_sd_df = weight_mean_sd_df.sort_values(by = ['cv'])

print('\nlow cv weights words: \n')
print(weight_mean_sd_df.head(10))
print('\nhigh cv weights words: \n')
print(weight_mean_sd_df.tail(10))


##############################################################
# Saving files to upload
##############################################################

# Saving the files needed for submission
final_model = my_neural_network

# Creating Directory
if not os.path.exists(root_dir + '/to_submit'):
    os.makedirs(root_dir + '/to_submit')

# Changing working directory
os.chdir(root_dir + '/to_submit')

# Saving the model as a dict
torch.save(final_model.state_dict(), "./part3_state.chkpt")

# Saving Accuracy on train and test set in a new text file
with open('project1_part3_results.txt','w+') as file:
    file.write(str(final_model.train_accuracy[-1]))
    file.write('\n')
    file.write(str(final_model.test_accuracy[-1]))
    file.close()

# Saving plots
final_model.generate_training_pictures()
final_model.fig1.savefig('project1_part3_plots_loss.png')
final_model.fig2.savefig('project1_part3_plots_accuracy.png')