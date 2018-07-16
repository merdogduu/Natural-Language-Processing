from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import PorterStemmer
from urllib.request import urlopen
from nltk import ngrams
import nltk
import re, string


a  = open('beatsreview.csv')

new_text=a.read()

ps=PorterStemmer()## kelime ayırmaca
##DELETE PUNCTUATIONS
regex = re.compile('[%s]' % re.escape(string.punctuation))
new_text = regex.sub('', new_text)
##

words=word_tokenize(new_text) ##cümledeki kelimeleri çekiyo
newarray=[]

#stem words

for w in words:
    newarray.append(ps.stem(w))##array'e ekliyo kelimeleri köklerine ayırıp.
     
#print(newarray)
    
##  stop words
    
stop_words=set(stopwords.words("english"))

words=word_tokenize(new_text)
filtered_sentence=[]
for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
  
print(filtered_sentence)

makeitastring = ' '.join(map(str, filtered_sentence))## array convert to String


#NUMBER OF WORDS

word_counts=Counter(filtered_sentence)
number_of_words=word_counts.most_common(len(filtered_sentence))

print(number_of_words)

##PHASE2!!!!!!

## GROUP WORDSSS

group_number=int(input("Give a number")) ## users give a number and we convert to int
sixgrams = ngrams(new_text.split(), group_number) ##girdiğimiz input kadar yazıp sonra split ediyo.
for grams in sixgrams:
	print (grams)

## FREQUENCE AND GROUPSSS

frequence=int(input("Enter a number for frequence: " ))

bigram_freq = {}		
length = len(filtered_sentence)
n=int(input("best result: " ))
for i in range(length-1):
    bigram = (filtered_sentence[i], filtered_sentence[i+1]) ##arrayin içindekileri alıp ikili grup haline getiriyo.
    if bigram not in bigram_freq:  						    ## ikili grupla eşleşen aynı bi grup varsa sayıyo
        bigram_freq[bigram] = 0
    bigram_freq[bigram] += 1

new_array=[ ]
for bigram in bigram_freq:
	if bigram_freq[bigram] == frequence:		##hangi frequence'i girersek o kadar tekrar edenleri bastırıyo
		new_array.append(bigram)
		if len(new_array)>=n:					##En fazla kaç tane bastırmak istediğimizi hesaplıyo.
			break
print(new_array)								##Print ediyo 

##SCORE

import nltk.collocations
import nltk.corpus
import collections
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()		
finder = BigramCollocationFinder.from_words(filtered_sentence)	
#finder.apply_freq_filter(3)
scored = finder.score_ngrams(bigram_measures.raw_freq)			
##sorted(bigram for bigram, score in scored).  					##funtion sorted
print(scored)


POSTag=nltk.pos_tag(words)
print (POSTag)     ##nltk.pos_tag helps us to pos tagging



def numOfTags():                   
    mostcommon=int(input("Enter a number for most common POS Tag: "))
    pos_counts = collections.Counter((subl[1] for subl in POSTag))
    print (pos_counts.most_common(mostcommon))
numOfTags()

name=input("Which part of speech do you want: ")
def findWords(POSTag, name):
    word_tag_fd = nltk.FreqDist(POSTag)
    print([wt[0] for (wt, _) in word_tag_fd.most_common() if wt[1] == name])


findWords(POSTag,name)





##Phase 4 **********

from nltk.corpus import movie_reviews
import random

nltk.download('movie_reviews')

documents = [(list(movie_reviews.words(fileid)),category)

             for category in movie_reviews.categories()

             for fileid in movie_reviews.fileids(category)]    



random.shuffle(documents)



#print(documents[1])



all_words = []



for w in movie_reviews.words():

    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)



word_features = list(all_words.keys())[:3000]



def find_features (document):

    words = set(document)

    features= {}



    for w in word_features:

        features[w] = (w in words)



    return features



#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))



featuresets =[(find_features(rev), category) for (rev, category) in documents]



training_set = featuresets[:1900]

testing_set = featuresets[1900:]



classifier = nltk.NaiveBayesClassifier.train(training_set)



#classifier_f= open("naivebayes.pickle","rb")

#classifier= pickle.load(classifier_f)

#classifier_f.close()





print("Naive Bayes accuracy percent:", (nltk.classify.accuracy(classifier,testing_set))*100)

classifier.show_most_informative_features(15)



#import pickle

#save_classifier= open ("naivebayes.pickle","wb")

#pickle.dump(classifier, save_classifier)



#save_classifier.close()


