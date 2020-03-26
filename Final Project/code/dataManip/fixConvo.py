import pickle as pk
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import numpy as np
import nltk
import pdb
threshold = 10
trainSplit = 0.75
maxSentenceLength=10

def nonTossSentences(smallDataSize=-1):
    with open("./trainDataINv110.pk","rb") as f:
        data = pk.load(f)

    smallData = []
    for d in data:
        if ('<toss>' in d or len(d.split())>maxSentenceLength):
            pass
        else:
            smallData.append(d)
    nd = np.array(smallData)
    if(smallDataSize>0):
        nd = nd[0:smallDataSize]
    print(nd.shape)
    din = nd[:-1]
    dout = nd[1:]
    if(smallDataSize<=0):
        with open('../data/nontosstrain'+str(sentenceLengthThresh)+'IN'+str(threshold)+'.pk','wb') as f:
            pk.dump(din,f)
        with open('../data/nontosstrain'+str(sentenceLengthThresh)+'OUT'+str(threshold)+'.pk','wb') as f:
            pk.dump(dout,f)
    else:
        with open('../data/nontossSMALLTRAIN'+str(sentenceLengthThresh)+'IN'+str(threshold)+'.pk','wb') as f:
            pk.dump(din,f)
        with open('../data/nontossSMALLTRAIN'+str(sentenceLengthThresh)+'OUT'+str(threshold)+'.pk','wb') as f:
            pk.dump(dout,f)

def nltkData(addUser=False):
    with open('./vocab.pk','rb') as f:
        vocab = pk.load(f)

    with open('./conversation.pk','rb') as f:
        convo = pk.load(f)

    with open('./userOrder.pk','rb') as f:
        userOrder = pk.load(f)


    with open('./maintains'+str(threshold)+'.pk','rb') as f:
        mtn = pk.load(f)

    with open('./throwaways'+str(threshold)+'.pk','rb') as f:
        toss = pk.load(f)

    with open('./fixers'+str(threshold)+'.pk','rb') as f:
        fix = pk.load(f)

    mtnChecker = []
    fixChecker = {}
    for pos in fix:
        if(pos[4]=='standard'):
            fixChecker[pos[0]]=pos[1]
        else:
            print("does this happen?")
            fixChecker[pos[1]]=pos[0]

    for pos in mtn:
        mtnChecker.append(pos[0])
        #fixChecker.append(pos[])

    for x in convo:
        for w in x.split():
            assert(w in vocab.keys())

    newConvo = []
    posTagMaps={}
    newInputs=[]
    newOutputs=[]
    userCounter=0
    numAdded=2
    pdb.set_trace()

    for x in convo:
        outString = "<start> "
        if(addUser):
            inString="<start> " + " <"+str(userOrder[userCounter]).replace(" ","")+"> "
        else:
            inString=outString
        counter=0
        POSs = nltk.pos_tag(x.split())
        for w in x.split():
            if(counter>=maxSentenceLength-numAdded):
                break
            if(w in toss):
                #print("have we tossed?")
                tag = POSs[counter][1]
                tPOS = "<"+tag+">"
                inString+=" <"+tag+"> "
                outString+=" <"+tag+"> "
                if(tPOS in posTagMaps.keys()):
                    posTagMaps[tPOS].add(w)
                else:
                    posTagMaps[tPOS]= set(w)
            elif(w in fixChecker.keys()):
                inString+=" "+fixChecker[w]+" "
                outString+=" "+fixChecker[w]+" "
            elif(w in mtnChecker or w in fixChecker.values()):
                #print("MAINTAIN:"+w)
                inString+=" "+w+" "
                outString+=" "+w+" "
            else:
                print("plz")
            counter+=1
        outString+=" <end>"
        finalInString=""
        tempCounter=0
        if(len(inString.split())>=maxSentenceLength):
            for x in inString.split():
                if(tempCounter>=maxSentenceLength-numAdded):
                    break
                else:
                    finalInString+=x+" "
            finalInString+=" <end>"
        else:
            finalInString=inString+" <end>"
        if(finalInString.split()[0]!='<start>'):
            pdb.set_trace()
        userCounter+=1
        newInputs.append(finalInString)
        newOutputs.append(outString)
        #newConvo.append(newString)
    data = newInputs[:-1]
    labels=newOutputs[1:]
    pdb.set_trace()
    #data = np.array(newConvo[:-1])
    #labels = np.array(newConvo[1:])

    dataSize = len(data)
    #indices = np.arange(dataSize).astype(int)
    #np.random.shuffle(indices)
    #indices = indices.astype(int)
    #print(indices[0])
    #data = data[indices]
    #labels = labels[indices]
    num_validation_samples = int((1-trainSplit) * dataSize)

    #limitedTokenizer = Tokenizer().fit_on_texts(dialogue)


    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    if(addUser):
        addUsertxt="USERLB"
    else:
        addUsertxt=""
    #pdb.set_trace()
    with open('./POSDICT'+str(threshold)+'.pk','wb') as f:
        pk.dump(posTagMaps,f)
    with open('./trainDataINPOSv1'+str(threshold)+str(addUsertxt)+'len'+str(maxSentenceLength)+'.pk','wb') as f:
        pk.dump(x_train,f)
    with open('./trainDataOUTPOSv1'+str(threshold)+str(addUsertxt)+'len'+str(maxSentenceLength)+'.pk','wb') as f:
        pk.dump(y_train,f)
    with open('./valDataINPOSv1'+str(threshold)+str(addUsertxt)+'len'+str(maxSentenceLength)+'.pk','wb') as f:
        pk.dump(x_val,f)
    with open('./valDataOUTPOSv1'+str(threshold)+str(addUsertxt)+'len'+str(maxSentenceLength)+'.pk','wb') as f:
        pk.dump(y_val,f)


def createTrainTest(addUser=False):
    with open('./vocab.pk','rb') as f:
        vocab = pk.load(f)

    with open('./conversation.pk','rb') as f:
        convo = pk.load(f)

    with open('./userOrder.pk','rb') as f:
        userOrder = pk.load(f)


    with open('./maintains'+str(threshold)+'.pk','rb') as f:
        mtn = pk.load(f)

    with open('./throwaways'+str(threshold)+'.pk','rb') as f:
        toss = pk.load(f)

    with open('./fixers'+str(threshold)+'.pk','rb') as f:
        fix = pk.load(f)

    mtnChecker = []
    fixChecker = {}
    for pos in fix:
        if(pos[4]=='standard'):
            fixChecker[pos[0]]=pos[1]
        else:
            print("does this happen?")
            fixChecker[pos[1]]=pos[0]

    for pos in mtn:
        mtnChecker.append(pos[0])
        #fixChecker.append(pos[])

    for x in convo:
        for w in x.split():
            assert(w in vocab.keys())

    newConvo = []
    userCounter = 0
    if(addUser):
        stringOffset = 2 +1
    else:
        stringOffset = 2
    for x in convo:
        newString = "<start> "
        counter=0
        for w in x.split():
            if(counter>=maxSentenceLength-stringOffset):
                break
            if(w in toss):
                #print("have we tossed?")
                newString+=" <toss> "
            elif(w in fixChecker.keys()):
                newString+=" "+fixChecker[w]+" "
            elif(w in mtnChecker or w in fixChecker.values()):
                #print("MAINTAIN:"+w)
                newString+=" "+w+" "
            else:
                print("plz")
            counter+=1
        newString+=" <end>"
        userCounter+=1
        newConvo.append(newString)
    data = np.array(newConvo[:-1])
    labels = np.array(newConvo[1:])

    dataSize = len(data)
    #indices = np.arange(dataSize).astype(int)
    #np.random.shuffle(indices)
    #indices = indices.astype(int)
    #print(indices[0])
    #data = data[indices]
    #labels = labels[indices]
    num_validation_samples = int(trainSplit * dataSize)

    #limitedTokenizer = Tokenizer().fit_on_texts(dialogue)


    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    with open('./trainDataINv1'+str(threshold)+'.pk','wb') as f:
        pk.dump(x_train,f)
    with open('./trainDataOUTv1'+str(threshold)+'.pk','wb') as f:
        pk.dump(y_train,f)
    with open('./valDataINv1'+str(threshold)+'.pk','wb') as f:
        pk.dump(x_val,f)
    with open('./valDataOUTv1'+str(threshold)+'.pk','wb') as f:
        pk.dump(y_val,f)
nltkData(True)
#createTrainTest()
#nonTossSentences(100)
