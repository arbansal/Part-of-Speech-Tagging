#!/usr/bin/python2
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Mahesh Latnekar, Chaitanya Patil, Arpit Bansal
#          (arbansal-mrlatnek-chpatil)
# (based on skeleton code by D. Crandall, Oct 2018)
#
# (1)Approach: We have first calculated the Prior and Transition probabilities from our training data file (bc.train from Part 1) and used a dictionary to store the Prior probabilties and a dictionary of dictionaries for storing the Transition probabilties.

#For the SIMPLIFIED ALGORITHM, we calculate the max of P(Oi | li) ie. and the Prior probabiities P(li)
#P(li | li-1) ie.probablity of an given letter transitioning to its subsequent letter (eg. in word train, transition of t to r) and \
# For the VITERBI ALGORITHM, we multiply the Initial and Emission prob for the first state and then multiply the latter with the transition probabilties, finding max at each stage. Then we use backtracking to confirm the sequence.

#(2) Description of how program works,



#We have created dictionary called 'prior1' which stores probablity of a letter being the first letter in the words contained
#in training file ('bc.train').
#'prior' dictionary stores the probablity of the occurence of letter in the entire training file.
#Using the same file, we calculated the probability of a letter transitioning to any other letter (in TRAIN_LETTER) and store in 
#a dictionary of dictionaries 'transition' which is indexed by the character value.
#
# The emission probabilities have been calculated by comparing The grid of the test character and approximating the number of pixels
#matched between the train_letter and the test_letter read by the program.
# If there is a matching '*' between the 2, we multiply the probability by 0.9 and if not we multiply it by 0.1.  
# For a matching ' ', we multiple the probability by 0.7 and if not, we multiply it by 0.3.(A matching '*' is given more importance than a matching ' ').
#We expiremented various times assigning different probablities to '*' and " ", and found that setting the probablity of 0.9 to matching "*" and setting probablity of 0.7 to matching " " yeilded more correct answers.
# Now after finding all the required probablities we implemented Simple and Viterbi Algorithm
#Simple approach: Uses 'simplified' function to evaluate. This function takes train and test letters as arguments and returns the letter
#with highest of values obtained by multiping emission and prior probablities.
#Viterbi Algorithm: 'viterbi1' dictionary  stores the maximum probablity probablity and 'viterbi2' dictionary is used for backtracking.
#values in 'viterbi1' are calculated as follows:
#for the 1st letter it stores : Initial_Probablity * Emission Probablity
#from 2nd letter it stores : max{ Probablity at [i-1] position * Emission Probablity * Transition Probablity}




from PIL import Image, ImageDraw, ImageFont
import sys
import math
from collections import defaultdict
import numpy as np
CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
#    print im.size
#    print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
#    print(type(TRAIN_LETTERS))
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]

    return exemplars

#####
# main program
(train_img_fname,train_txt_fname,test_img_fname) = sys.argv[1:] 
train_letters = load_training_letters(train_img_fname)

test_letters = load_letters(test_img_fname)

train_data = read_data(train_txt_fname)

#print(train_letters['a'][0])
#print(len(train_letters['a'])) #25
#print(len(train_letters['a'][0])) #14

#all_letters = []
#
#for (sentence,tags) in train_data:
#    for word in sentence:
#        for letters in word:
#            all_letters.append(letters)
#
#all_letters_set = set(all_letters)
#print(all_letters_set)

##########################################################################
# Creating dictionary of dictionaries for Prior or Initial probabilites

prior = defaultdict(int)

for letter in train_letters:
    prior[letter] = 0
    prior['count'] = 0

max_len = 0 #max length of a word

for (sentence,pos) in train_data:
    for word in sentence:
        if(len(word)>max_len):
            max_len = len(word)

prior1 = defaultdict(dict)
            
for i in range(max_len):
    prior1[str(i)] = defaultdict(int)

#Initial Probabilities : probabilities of a character being the first letter of a statement.
    
#def getInitialProb(train_data, train_letters):
    
for (sentence,pos) in train_data:
    for word in sentence:
        for i in range(len(word)):
                prior[word[i]] +=1
                prior['count'] +=1
    
for letter in train_letters:
    num = prior[letter]/float(prior['count'])
    if (num>0):
        prior['P_' + letter] = -math.log(num)

for (sentence,pos) in train_data:
    for word in sentence:
        for i in range(len(word)):
            prior1[str(i)][word[i]] +=1
            prior1[str(i)]['count'] +=1
    
for keys in prior1:
    for letter in train_letters:
        num = prior1[keys][letter]/float(prior1[keys]['count'])
        if(num>0):
            prior1[keys]['P_' + letter] = -math.log(num)

#print(prior['a'], prior['count'], prior['P_a'])
#print(prior1['0']['a'], prior1['0']['count'], prior1['0']['P_a'])
#return(prior, prior1)

###############################################################################
#Transition probabilities: probability of transitioning from one character to another

transition = defaultdict(dict)
for letter in train_letters:
    transition[letter] = defaultdict(int) 

#def getTransitionProb(train_data):
char_list = []

for letter in train_letters:
    char_list.append(letter)
#print(len(char_list))
    
for char1 in char_list: 
    for char2 in char_list:   
        transition[char1][char2] = 0
        transition[char1]['count'] = 0

for (sentence,pos) in train_data:
    for word in sentence:       
        for i in range(len(word)-1):
            if ((word[i] in char_list) and (word[i+1] in char_list)):
#            if(word[i+1] not in transition[word[i]]):
#                transition[word[i]][word[i+1]] = 1  
#            else:
                transition[word[i]][word[i+1]] += 1
                transition[word[i]]['count'] += 1
#            print(word[i])
#        break
        
for keys in transition:
    for letter in train_letters:
        if transition[keys]['count'] == 0:
            transition[keys]['P_' + letter] =-math.log(0.000001)
        else:
            num = transition[keys][letter]/float(transition[keys]['count'])
            if (num>0):
                transition[keys]['P_' + letter] = -math.log(num)

#print(transition['a']['n'], transition['a']['count'], transition['a']['P_n'])
#print(transition)    
#    print(transition['v'])
            
#    return(transition)
############################################################################################################3
#probabilities of the test character grid representing a particular character
#The grid of the test character was matched against the grid of the train letters.

#def getEmissionProb(train_data, train_letters, test_letters): 
            
emission = defaultdict(dict)
for i in range(len(test_letters)):
    emission[str(i)] = defaultdict(int) 
    
for i in range(len(test_letters)):
    for letters in train_letters:
        emission[str(i)][letters] = 0            
                   
for i in range(len(test_letters)):
    for letters in train_letters:
        matches = 1
        num_train = 0
        num_test = 0
        for height in range(0, CHARACTER_HEIGHT):
            for width in range(0, CHARACTER_WIDTH):

                if (train_letters[letters][height][width] == test_letters[i][height][width]) :
                    if (train_letters[letters][height][width] == '*'):
                        matches *= 0.9
                    else:
                        matches *=0.7
                else:
                    if(train_letters[letters][height][width] == '*'):
                        matches *= 0.3
                    else:
                        matches *= 0.1
        emission[str(i)][letters] = -math.log(matches)


def simplified(train_letters, test_letters):
                    
    sentence = []
    for i in range(len(test_letters)):
        probablity_list=[]
        #print('i word',sentence[i])
        for letter in char_list:                 
            probablity_list.append(emission[str(i)][letter] + prior['P_' + letter])
        tag  = char_list[probablity_list.index(min(probablity_list))]
        sentence.append(tag)
    return sentence

def viterbi(train_letters, test_letters):
    sentence = []
    
    viterbi_1 = defaultdict(dict)
    for i in range(len(test_letters)):
        viterbi_1[str(i)] = defaultdict(int)
    
#    for i in range(len(test_letters)):
#        for letter in char_list:            
#            viterbi_1[str(i)][letter] = 0
    viterbi_2 = defaultdict(dict)
    for i in range(1,len(test_letters)):
        viterbi_2[str(i)] = defaultdict(int)

    for letter in char_list:
        viterbi_1['0'][letter] = (emission['0'][letter] + prior1['0']['P_' + letter])#*prior????
       # viterbi_1['0'][letter] = (emission['0'][letter])
    #print(viterbi_1['0'])

    for i in range(1,len(test_letters)):            
        for letter1 in char_list:
            templist = []
            for letter2 in char_list:
                emis = emission[str(i)][letter1 ]
                trans = transition[letter2]['P_' + letter1]
                temp = viterbi_1[str(i-1)][letter2]+emis+trans #*emis
                templist.append(temp)                
            max_pos = char_list[templist.index(min(templist))]                
            viterbi_1[str(i)][letter1] = min(templist)
            viterbi_2[str(i)][letter1] = max_pos
#    print(viterbi_1['1'])
    last_col=[]       
    for letter in char_list:
        last_col.append(viterbi_1[str(len(test_letters)-1)][letter])
    last_pos = char_list[last_col.index(min(last_col))]
    sentence.append(last_pos)
    #temp = viterbi_2[str(length(sentence)-1)][last_word]
    temp = last_pos
    #print(viterbi_1['3'])
    for i in range(len(test_letters)-1,0,-1):
        temp = viterbi_2[str(i)][temp]
        #print(i,temp)
        
        sentence.append(temp)
    #print(partOfSpeech)
    sentence = list(reversed(sentence))
    return sentence

result1 = simplified(train_letters, test_letters)
result2 = viterbi(train_letters, test_letters)
#print(result1) 
#print(result2)
res1 = ''.join(result1)
print "Simple: ", res1
res2 = ''.join(result2)
print "Viterbi: ", res2
print "Final answer:"
print res2  
                   
## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!                
                
# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
#print "\n".join([ r for r in train_letters['a'] ])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
#print "\n".join([ r for r in test_letters[2] ])
#print(test_letters)
