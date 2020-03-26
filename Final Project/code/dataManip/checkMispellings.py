import pickle as pk
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import numpy as np
import pdb


def getVeryCommonWords(threshold,vocab,speller):
    commonWords = []
    for v in vocab.keys():
        if(vocab[v]>threshold):
            speller.word_frequency.dictionary[v] = vocab[v]
    speller.word_frequency._update_dictionary()
    return speller

def checkWordLengths():
    threshold=10
    with open('./maintains'+str(threshold)+'.pk','rb') as f:
        mtn = pk.load(f)

    with open('./throwaways'+str(threshold)+'.pk','rb') as f:
        toss = pk.load(f)

    with open('./fixers'+str(threshold)+'.pk','rb') as f:
        fix = pk.load(f)

    bins=[1000,100,90,80,70,60,50,40,30,20,10]
    newHeights=np.zeros(len(bins))
    pdb.set_trace()
    '''for x in list(mtn.values()):
        tempCounter=0
        while(tempCounter<len(bins)):
            if(tempCounter==0):
                if(x>=bins[tempCounter]):
                    newHeights[tempCounter]+=wordFrequency[x]
            elif(tempCounter==len(bins)-1):
                if(x>=bins[tempCounter]):
                    newHeights[tempCounter]+=wordFrequency[x]
            else:
                if(x<bins[tempCounter-1] and x >=bins[tempCounter]):
                    newHeights[tempCounter]+=wordFrequency[x]
            tempCounter+=1
    for x in list(fix.values()):
        tempCounter=0
        while(tempCounter<len(bins)):
            if(tempCounter==0):
                if(x>=bins[tempCounter]):
                    newHeights[tempCounter]+=wordFrequency[x]
            elif(tempCounter==len(bins)-1):
                if(x>=bins[tempCounter]):
                    newHeights[tempCounter]+=wordFrequency[x]
            else:
                if(x<bins[tempCounter-1] and x >=bins[tempCounter]):
                    newHeights[tempCounter]+=wordFrequency[x]
            tempCounter+=1

    pdb.set_trace()
    plt.bar(x=np.arange(0,10,1),height=newHeights) #height=10*((np.array(list(wordFrequency.values()))/10).astype(int)))
    plt.savefig('./wordFrequency')
    plt.title("frequency of words (magnitude of 10s) w/out error check")
    plt.clf()'''
def alternateCheck(vocab,threshold):
    speller = SpellChecker(distance=1)
    spell = getVeryCommonWords(threshold,vocab,speller)
    maintainWords=set()
    mightbewrong=set()
    throwawaywords=set()
    wrongToCorrMappings=set()
    for v in vocab.keys():
        correction = spell.correction(v)
        if(v != correction):
            if(correction in vocab.keys()):
                if(vocab[v]<threshold and vocab[correction]>threshold):
                    wrongToCorrMappings.add((v,correction,vocab[v],vocab[correction],'standard'))
                elif(vocab[v]>threshold and vocab[correction]<threshold):
                    wrongToCorrMappings.add((correction,v,vocab[v],vocab[correction],'flip'))
                elif(vocab[v]<threshold and vocab[correction]<threshold):
                    throwawaywords.add(v)
                else:
                    maintainWords.add((v,vocab[v]))
            else:
                if(vocab[v]>threshold):
                    maintainWords.add((v,vocab[v]))
                else:
                    throwawaywords.add(v)
        else:
            if(vocab[v]<threshold):
                throwawaywords.add(v)
            else:
                maintainWords.add((v,vocab[v]))

    print("Words i am gonna keep")
    print(maintainWords)
    print("Words im gonna just toss")
    print(throwawaywords)
    print("Words im gonna fix")
    print(wrongToCorrMappings)

    with open('./maintains'+str(threshold)+'.pk','wb') as f:
        pk.dump(maintainWords,f)
    with open('./throwaways'+str(threshold)+'.pk','wb') as f:
        pk.dump(throwawaywords,f)
    with open('./fixers'+str(threshold)+'.pk','wb') as f:
        pk.dump(wrongToCorrMappings,f)




with open('./vocab.pk','rb') as f:
    vocab = pk.load(f)
setMisspelledWords=set()
matchingInDict = set()

mapping = {}
numOneoffs = 0
for e in vocab.values():
    if(e in mapping):
        mapping[e]+=1
    else:
        mapping[e]=1
    if(e<=2):
        numOneoffs+=1

print(numOneoffs/len(vocab))# half of our vocab is filled with oneoffs, 65% with two offs
checkWordLengths()
'''
for each word in our vocabulary, check if its spelled wrong.
If it isnt ,
    if(frequency < threshopkld):
        remove?
    else:
        keep
Else, check if its in our dictionary already and check if its under some thereshold of word_frequency
    if its correction is in the dictionary
        and the word is used less than the threshold,
            we will correct all instances of the word in the chat to it.
        word is NOT used less than the threshold
            leave it alone, but keep a list of these for future reference
    else (correction is not in dictionary)
        if(word used less than threshold):
            store a copy, throw away?
        else:
            stoe a copy and keep?
(Common mispellings are an issue, but maybe not?) E.g. maybe its good for our model to learn when we use 'the' versus 'teh'
'''
#for i in [3,10]:
#    alternateCheck(vocab,i)
#alternateCheck(vocab,)
#print(mapping.values())
#print(np.unique(vocab.values()))
#plt.hist(vocab.values())
#plt.show()
'''
counter = 0
denom=0
for word in vocab.keys():
    if(vocab[word]<2):
        fixed = spell.correction(word)
        if(fixed != word):
            setMisspelledWords.add((word,fixed))
            if(fixed in vocab.keys()):
                matchingInDict.add((w))
            counter+=1
        denom+=1

print(counter/denom)


with open('./mismatches.pk','wb') as f:
    pk.dump(setMisspelledWords,f)
'''
#spell =  SpellChecker()
#spell.word_frequency.load_words(['yall'])
#correctedWord = spell.correction(holdit)
#if(correctedWord != holdit or len(spell.candidates(holdit))>1):
#    pass
    #pdb.set_trace() #yall correct to all
#    setMisspelledWords.add(holdit)
