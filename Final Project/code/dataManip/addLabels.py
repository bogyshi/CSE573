import numpy as np
import pickle as pk
import pdb
import os
import random

thresh=10

def exploreLabels():
    txtPath = '../data/toConvert/'
    labelFile='./POSDICT'+str(thresh)+'.pk'
    with open(labelFile,'rb') as f:
        substitutes = pk.load(f)
    tolower = lambda a: a.lower()
    labels = list(substitutes.keys())
    templabels = []
    for x in labels: templabels.append(tolower(x))
    labels=templabels
    for f in os.listdir('../data/toConvert/'):
        with open(txtPath+f,'r') as txt:
            print(txt)
            counter=0
            actualOut=[]
            inputs=[]
            predTrans=[]
            for r in txt.readlines():
                if(counter<6):
                    print(r)
                if(counter%5==2):
                    newString=""
                    for word in r.split():
                        if(word in labels):
                            tempList = list(substitutes[word.upper()])
                            index = int(random.random()*len(tempList))
                            #pdb.set_trace()
                            newString+=' ' +tempList[index]+ ' '
                        else:
                            newString+=' '+word+' '
                    predTrans.append(newString)
                elif(counter%5==1):
                    inputs.append(r)
                elif(counter%5==3):
                    actualOut.append(r)
                counter+=1
        with open('../data/converted/'+str(f),'w') as outfile:
            for ins in zip(inputs,predTrans):
                outfile.write((ins[0]+ins[1]+'\n'))
exploreLabels()
