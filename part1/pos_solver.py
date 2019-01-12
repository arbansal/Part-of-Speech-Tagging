#!/usr/bin/python2
####################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids: Mahesh Latnekar, Chaitanya Patil, Arpit Bansal
# Authors: arbansal-mrlatnek-chpatil
# (Based on skeleton code by D. Crandall)
#
## OUTPUT:
#
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct: 
#   0. Ground truth:      100.00%              100.00%
#         1. Simple:       93.95%               47.50%
#            2. HMM:       94.96%               54.45%
#        3. Complex:       94.19%               51.75%

# How have we represented the problem?
#
#We have mapped this problem into a Hidden Markov model wherein we consider each test word in the test sentence as an Observed state and then try to compute the most likely part of speech tag for the observed state (which are the hidden states). 
#
#The model requires calculating 3 probabilities:
#Initial Probabilities: Probability of a part of speech being the first word of a sentence.
#Transition Probabilities: Probability of transitioning from one part of speech to another
#Emission Probabilities: Probability of the a word representing a particular part of speech
#
## How we did part of speech tagging?
#
#We have first calculated the initial probabilities, the transition probabilities and emission probabilities based on our training dataset present in 'bc.train' file. 
#
#Initial probabilities are stored in a dictionary 'prior' which calculates the probability of a part of speech in the entire training set.
#We also have utilized another dictionary (dictionary of dictionaries) 'prior1' which calculates the probability of a part of speech occurring at a particular position in the sentence for the given training set. (For computing initial probabilities, we traversed the training file and counted the first words of the sentence, and then normalized these probabilities.)
#
#Transition probabilities are stored in dictionary (dictionary of dictionaries) 'transition' which calculates the probabilities of one part of speech following another. (These values have also been calculated from the training file wherein we have traversed all the words and parts of speech tags and counted the occurrences of different parts of speech tags following another tag.)
#
#Emission probabilities are stored in dictionary (dictionary of dictionaries) and for this we have traversed the training file and probabilities of a word occuring in a particular part of speech. 
#
#To handle zero probabilities, we have chosen a value of 0.0000000001 which is assigned in the probability value instead of 0. This value has been derived after performing a number of tests on different arbitrary values. 
#
#For the Simplified algorithm we calculate the max of P(W|S)*P(S) for each position in the sentence.
#
#For the Viterbi algorithm, we have utilized 2 dictioneries, viterbi_1 for calculating intermediate values (which serves as memoization table) and viterbi_2 for backtracking.
#For position 0, we multiply the emission probability with Initial probability (position '0' in our prior1 dictionary).
#For the subsequent states, we multiply the transition probabilities of all parts of speech and take the maximum value and store it.   
#values in 'viterbi1' are calculated as follows:
#for the 1st letter it stores : Initial_Probablity * Emission Probablity
#from 2nd letter it stores : max{ Probablity at [i-1] position * Emission Probablity * Transition Probablity}

#For the Complex algorithm, we initially start with a random sample with random values for POS given length of sentence and take 1000 iterations. 
#The first 25 iterations are for the warmup period and we store the subsequent 975 values. We iterate through each POS and take a final value for which the frequence was the highest.
#This happens for all POS in the length of the sentence and after 1000 samples the final pattern recognized is returned.
####

import random 
import math
import numpy as np
from collections import Counter
from collections import defaultdict

class Solver:
    # To calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. 
    def posterior(self, model, sentence, label):
               
        if model == "Simple":
            posterior = 0
            for i in range(len(sentence)):
                
                #print('i word',sentence[i])
                              
                posterior += math.log((self.emmission[label[i]]['P_' + sentence[i]] )*(self.prior['P_' + label[i]]))
            return posterior
        elif model == "Complex":
            posterior = 0
            em_p = 0
            dtr_p = 0
            if len(sentence) == 1:
                posterior = math.log((self.emmission[label[0]]['P_' + sentence[0]] )*(self.prior['P_' + label[0]]))
            else:
                for i in range(0,len(sentence)):
                    em_p += math.log(self.emmission[label[i]]['P_' + sentence[i]])
                for j in range(0,len(sentence)-2):    
                    dtr_p += math.log(self.dtransition[label[j] + '_' + label[j+1]]['P_' + label[j+2]])
                initial_p = math.log(self.prior1['0']['P_' + label[0]])
                transition_1_p = math.log(self.transition[label[0]]['P_' + label[1]])
                posterior = em_p + dtr_p + initial_p + transition_1_p 
            return posterior
        elif model == "HMM":
            posterior = 0
            em_p = 0
            tr_p = 0
            for i in range(0,len(sentence)):
                em_p += math.log(self.emmission[label[i]]['P_' + sentence[i]])
            for j in range(0,len(sentence)-1):
                tr_p += math.log(self.transition[label[j]]['P_' + label[j+1]])
            initial_p = math.log(self.prior1['0']['P_' + label[0]])
            posterior = em_p + tr_p + initial_p
            return posterior
            
        else:
            print("Unknown algo!")

    # Training given data
    def train(self, train_data):
        pos_list = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb','x', '.']
        all_words=[]
        p_os = []
        
        # all_words contains list of all words, p_os contains all parts of speech in the train_data
        for (sentence,tags) in train_data:
            for word in sentence:
                all_words.append(word)
            for pos in tags:
                p_os.append(pos)
        
        #Dictionary of dictionary 'prior' for prior prob Si
        #counts occurence of each POS at a particular position in the sentence, 
        #stores with key value as the position of POS in the sentence

        prior1 = defaultdict(dict)
        prior = defaultdict(int)
        max_len = 0        
        for (sentence,pos) in train_data:
            if(len(sentence)>max_len):
                max_len = len(sentence)
     
        for i in range(max_len):
            prior1[str(i)] = defaultdict(int)

        for (sentence,pos) in train_data:
            for i in range(len(sentence)):
                prior1[str(i)][pos[i]] +=1
                prior1[str(i)]['count'] +=1
                
        for keys in prior1:
            for pos in pos_list:
                prior1[keys]['P_' + pos] = prior1[keys][pos]/float(prior1[keys]['count'])
                        
        for (sentence,pos) in train_data:
            for pos1 in pos:
                prior[pos1] +=1
                prior['count'] +=1
        
        for pos in pos_list:
            prior['P_' + pos] = prior[pos]/float(prior['count'])
            
        emmission = defaultdict(dict)
        transition = defaultdict(dict)
        for pos in p_os:
            emmission[pos] = defaultdict(int)
            transition[pos] = defaultdict(int)            

        #emission POS P(Wi/Si) for each word corresponding to each POS (eg. emission of 'poet' in 'nouns')
        
        for (word, pos) in zip(all_words,p_os):
            emmission[pos][word] +=1
            emmission[pos]['count'] += 1

        for (word, pos) in zip(all_words,p_os):
            emmission[pos]['P_' + word] = emmission[pos][word]/float(emmission[pos]['count'])
          
        #transition is counting the number of transitions from POS1 - POS2
        #Calculates prob of [S(i+1)| Si]    
        
        for x in pos_list:
            transition[x]['count'] = 0
        
        for (sentence,pos) in train_data:    
            for i in range(len(pos)-1):
                if(pos[i+1] not in transition[pos[i]]):
                    transition[pos[i]][pos[i+1]] = 1  
                else:
                    transition[pos[i]][pos[i+1]] += 1
                transition[pos[i]]['count'] += 1

        for keys in transition:
            for pos in pos_list:
                if(transition[keys]['count'] == 0):
                    transition[keys]['P_' + pos] = 0.0000000001
                else:
                    transition[keys]['P_' + pos] = transition[keys][pos]/float(transition[keys]['count'])
        
        #dtransition counts the number of transitions from POS1_POS2 to POS3
        #Calculates probability of Si+2 | Si, Si+1        

        dtransition = defaultdict(dict)
        for pos1 in pos_list:
            for pos2 in pos_list:
                dtransition[pos1 + '_' + pos2] = defaultdict(int)
        
        for (word,pos) in train_data:
            for i in range(2,len(pos)):
                dtransition[pos[i-2] + '_' + pos[i-1]][pos[i]] += 1  
                dtransition[pos[i-2] + '_' + pos[i-1]]['count'] += 1

        for keys in dtransition:
            for pos in pos_list:
                if dtransition[keys]['count'] != 0:
                    if dtransition[keys][pos] != 0:
                        dtransition[keys]['P_' + pos] = dtransition[keys][pos]/float(dtransition[keys]['count'])
                    else:
                        dtransition[keys]['P_' + pos] = 0.0000000001
                else:
                    dtransition[keys]['P_' + pos] = 0.0000000001
        
        for keys in dtransition:
            for pos in pos_list:
                if dtransition[keys][pos] != 0:
                    dtransition[keys]['P_' + pos] = dtransition[keys][pos]/float(dtransition[keys]['count'])
                else:
                    dtransition[keys]['P_' + pos] = 0.0000000001
        
        self.all_words = all_words
        self.prior = prior
        self.prior1 = prior1
        self.transition = transition
        self.emmission = emmission
        self.dtransition = dtransition
        self.pos_list = pos_list
####################################################################################################

    # Functions for each algorithm. 
    def simplified(self, sentence):

        for word in sentence:
            for pos in self.pos_list:
                if word not in self.all_words:
                    #print('not there',word)
                    self.emmission[pos]['P_'+ word] = 0.0000000001
        
        for keys in self.emmission:
            a = []
            for new_keys in self.emmission[keys]:
                a.append(new_keys)
                
            for words in sentence:
                if words not in a:
                    #print('not there',words)
                    self.emmission[keys]['P_' + words] = 0.0000000001
                        
        partOfSpeech = []
        for i in range(len(sentence)):
            probablity_list=[]
            for pos in self.pos_list:                 
                probablity_list.append((self.emmission[pos]['P_' + sentence[i]] )*(self.prior['P_' + pos]))
            tag  = self.pos_list[probablity_list.index(max(probablity_list))]
            partOfSpeech.append(tag)
        
        return partOfSpeech

    def complex_mcmc(self, sentence):
        
        for word in sentence:
            for pos in self.pos_list:
                if word not in self.all_words:
                    self.emmission[pos]['P_'+ word] = 0.0000000000001
        
        for keys in self.emmission:
            a = []
            for new_keys in self.emmission[keys]:
                a.append(new_keys)
                
            for words in sentence:
                if words not in a:
                    self.emmission[keys]['P_' + words] = 0.0000000000000001
        majVote = defaultdict(dict)
        for i in range (len(sentence)):
            majVote[str(i)] = defaultdict(int)
        for i in range (len(sentence)):
            for pos in self.pos_list:
                majVote[str(i)][pos]=0
        
        sample = []
        for i in range(len(sentence)):
            temp = np.random.choice(self.pos_list)
            sample.append(temp)
            
        final_distribution = []
        for i in range(1000):
            if len(sentence) == 1:
                denominator = 0
                distribution = []
                probab = []
                for pos in self.pos_list: 
                    prior_val = self.prior1[str(0)]['P_' + sample[0]]
 #                       prior_val = self.prior['P_' + pos]
                    emiss = self.emmission[pos]['P_' + sentence[0]]
                    denominator += prior_val*emiss
                for pos in self.pos_list: 
                    prior_val = self.prior1[str(0)]['P_' + sample[0]]
 #                       prior_val = self.prior['P_' + pos]
                    emiss = self.emmission[pos]['P_' + sentence[0]]
                    numerator = prior_val*emiss
                    if denominator == 0:
                        probab.append(0)
                    else:
                        probab.append( numerator/float(denominator))
                if sum(probab)==0:
                    samp = np.random.choice(self.pos_list)
                else:
                    samp = np.random.choice(self.pos_list,p=probab)
                sample[0] = samp
            
                
            else:    
                for j in range(0,len(sentence)):
                    denominator = 0
                    distribution = []
                    probab = []
                    if(j==0):                
                        for pos in self.pos_list: 
                            prior_val = self.prior1[str(j)]['P_' + pos]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[pos][sample[j+1]]
                            if len(sentence) == 2:
                                d1 = 1
                            else:
                                d1 = self.dtransition[pos + '_' + sample[j+1]][sample[j+2]]
                            denominator += prior_val*emiss*trans*d1
                            
                        for pos in self.pos_list: 
                            prior_val = self.prior1[str(j)]['P_' + pos]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[pos][sample[j+1]]
                            if len(sentence) == 2:
                                d1 = 1
                            else:
                                d1 = self.dtransition[pos + '_' + sample[j+1]][sample[j+2]]
                            numerator = prior_val*emiss*trans*d1
                            if denominator == 0:
                                probab.append(0)
                            else:
                                probab.append( numerator/float(denominator))
                        if sum(probab)==0:
                            samp = np.random.choice(self.pos_list)
                        else:
                            samp = np.random.choice(self.pos_list,p=probab)
                        sample[j] = samp
                        majVote[str(j)][samp]+=1
                    
                    elif(j==1):
                        denominator = 0
                        probab = []
                        for pos in self.pos_list: 
                            #prior_val = self.prior1[str(j-1)]['P_' + sample[j-1]]
                            prior_val = self.prior['P_' + sample[j-1]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-1]][pos]
                            if len(sentence) == 2: # 
                                d1 = 1
                            else:
                                d1 = self.dtransition[sample[j-1] + '_' + pos][sample[j+1]]
                            if len(sentence)==3 or len(sentence)==2:
                                d2 = 1
                            else:
                                d2 = self.dtransition[pos + '_' + sample[j+1]][sample[j+2]]
                            denominator += prior_val*emiss*trans*d1*d2
                            
                        for pos in self.pos_list: 
                            #prior_val = self.prior1[str(j-1)]['P_' + sample[j-1]]
                            prior_val = self.prior['P_' + sample[j-1]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-1]][pos]
                            if len(sentence) == 2:
                                d1 = 1
                            else:
                                d1 = self.dtransition[sample[j-1] + '_' + pos][sample[j+1]]
                            if len(sentence)==3 or len(sentence)==2:
                                d2 = 1
                            else:
                                d2 = self.dtransition[pos + '_' + sample[j+1]][sample[j+2]]
                            numerator = prior_val*emiss*trans*d1*d2
                            if denominator == 0:
                                probab.append(0)
                            else:
                                probab.append( numerator/float(denominator))
                        if sum(probab)==0:
                            samp = np.random.choice(self.pos_list)
                        else:
                            samp = np.random.choice(self.pos_list,p=probab)
                        sample[j] = samp 
                        majVote[str(j)][samp]+=1
                    elif (j == len(sentence)-2): 
                        denominator = 0
                        probab = []
                        for pos in self.pos_list: 
                            #prior_val = self.prior1[str(j-2)]['P_' + sample[j-2]]
                            prior_val = self.prior['P_' + sample[j-2]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-2]][sample[j-1]]
                            d1 = self.dtransition[sample[j-2] + '_' + sample[j-1]][pos]
                            d2 = self.dtransition[sample[j-1] + '_' + pos][sample[j+1]]
                            denominator += prior_val*emiss*trans*d1*d2
                            
                        for pos in self.pos_list: 
                           # prior_val = self.prior1[str(j-2)]['P_' + sample[j-2]]
                            prior_val = self.prior['P_' + sample[j-2]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-2]][sample[j-1]]
                            d1 = self.dtransition[sample[j-2] +'_' +sample[j-1]][pos]
                            d2 = self.dtransition[sample[j-1] +'_' + pos][sample[j+1]]
                            numerator = prior_val*emiss*trans*d1*d2
                            if denominator == 0:
                                probab.append(0)
                            else:
                                probab.append( numerator/float(denominator))
                        if sum(probab)==0:
                            samp = np.random.choice(self.pos_list)
                        else:
                            samp = np.random.choice(self.pos_list,p=probab)
                        sample[j] = samp
                        majVote[str(j)][samp]+=1
                    elif (j == len(sentence)-1):
                        denominator = 0
                        probab = []
                        for pos in self.pos_list: 
                         #   prior_val = self.prior1[str(j-2)]['P_' + sample[j-2]]
                            prior_val = self.prior['P_' + sample[j-2]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-2]][sample[j-1]]
                            d1 = self.dtransition[sample[j-2] +'_' +sample[j-1]][pos]
                            denominator += prior_val*emiss*trans*d1
                            
                        for pos in self.pos_list: 
                            #prior_val = self.prior1[str(j-2)]['P_' + sample[j-2]]
                            prior_val = self.prior['P_' + sample[j-2]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-2]][sample[j-1]]
                            d1 = self.dtransition[sample[j-2] +'_' +sample[j-1]][pos]
                            numerator = prior_val*emiss*trans*d1
                            if denominator == 0:
                                probab.append(0)
                            else:
                                probab.append( numerator/float(denominator))
                        if sum(probab)==0:
                            samp = np.random.choice(self.pos_list)
                        else:
                            samp = np.random.choice(self.pos_list,p=probab)
                        sample[j] = samp
        #                    print(samp)
                        majVote[str(j)][samp]+=1
                    
                    
        
                    else: 
                        denominator = 0
                        probab = []
                        for pos in self.pos_list:
                            
                        #    prior_val = self.prior1[str(j-2)]['P_' + sample[j-2]]
                            prior_val = self.prior['P_' + sample[j-2]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-2]][sample[j-1]]
                            d1 = self.dtransition[sample[j-2] + '_' + sample[j-1]][pos]
                            d2 = self.dtransition[sample[j-1] + '_' + pos][sample[j+1]]
                            d3 = self.dtransition[pos +'_' + sample[j+1]][sample[j+2]]                    
                            denominator += prior_val*emiss*trans*d1*d2*d3
                        for pos in self.pos_list:
                        #    prior_val = self.prior1[str(j-2)]['P_' + sample[j-2]]
                            prior_val = self.prior['P_' + sample[j-2]]
                            emiss = self.emmission[pos]['P_' + sentence[j]]
                            trans = self.transition[sample[j-2]][sample[j-1]]
                            d1 = self.dtransition[sample[j-2] + '_' + sample[j-1]][pos]
                            d2 = self.dtransition[sample[j-1] + '_' + pos][sample[j+1]]
                            d3 = self.dtransition[pos + '_' + sample[j+1]][sample[j+2]]
                            numerator = prior_val*emiss*trans*d1*d2*d3
                            distribution.append(pos)
                            if denominator == 0:
                                probab.append(0)
                            else:
                                probab.append( numerator/float(denominator))
                        if sum(probab)==0:
                            samp = np.random.choice(self.pos_list)
                        else:
                            samp = np.random.choice(self.pos_list,p=probab)
                        sample[j] = samp
                        majVote[str(j)][samp]+=1
            if i > 25:
                final_distribution.append(','.join(sample))
        partOfSpeech = {dist:final_distribution.count(dist) for dist in set(final_distribution)}
        return max(partOfSpeech, key = partOfSpeech.get).split(',')
                            
        return partOfSpeech

    def hmm_viterbi(self, sentence):
        
        for word in sentence:
            for pos in self.pos_list:
                if word not in self.all_words:
                    self.emmission[pos]['P_'+ word] = 0.0000000000001
        
        for keys in self.emmission:
            a = []
            for new_keys in self.emmission[keys]:
                a.append(new_keys)
                
            for words in sentence:
                if words not in a:
                    self.emmission[keys]['P_' + words] = 0.0000000000000001
        
        partOfSpeech = []
        viterbi_1 = defaultdict(dict)
        for i in range(len(sentence)):
            viterbi_1[str(i)] = defaultdict(int)
        viterbi_2 = defaultdict(dict)
        for i in range(1,len(sentence)):
            viterbi_2[str(i)] = defaultdict(int)
        for pos in self.pos_list:
            viterbi_1['0'][pos] = (self.emmission[pos]['P_' + sentence[0]])*(self.prior1['0']['P_' + pos])#*prior????
        
        for i in range(1,len(sentence)):            
            for pos1 in self.pos_list:
                templist = []
                for pos2 in self.pos_list:
                    emmis = self.emmission[pos1]['P_' + sentence[i]]
                    temp = viterbi_1[str(i-1)][pos2]*(self.transition[pos2]['P_' + pos1])*emmis
                    templist.append(temp)                
                max_pos = self.pos_list[templist.index(max(templist))]                
                viterbi_1[str(i)][pos1] = max(templist)
                viterbi_2[str(i)][pos1] = max_pos
        last_col=[]       
        for pos in self.pos_list:
            last_col.append(viterbi_1[str(len(sentence)-1)][pos])
        last_pos=self.pos_list[last_col.index(max(last_col))]
        partOfSpeech.append(last_pos)
        #temp = viterbi_2[str(length(sentence)-1)][last_word]
        temp = last_pos
        for i in range(len(sentence)-1,0,-1):
            temp = viterbi_2[str(i)][temp]
            #print(i,temp)
            
            partOfSpeech.append(temp)
        #print(partOfSpeech)
        partOfSpeech = list(reversed(partOfSpeech))
        self.viterbi_1 = viterbi_1
        #print('viterbi',partOfSpeech)
        return partOfSpeech

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

