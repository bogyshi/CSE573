# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvals = {}
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if(state not in self.qvals.keys() or action not in self.qvals[state].keys()):
            return 0.0
        else:
            return self.qvals[state][action][3]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        bestASc = None
        possibleAs = []
        if(len(self.getLegalActions(state))==0 or state not in self.qvals.keys()):
            return 0.0
        for a in self.qvals[state]:
            possibleAs.append(a)
            qval = self.getQValue(state,a)
            if(bestASc is None or qval>bestASc):
                bestASc = qval
        if(bestASc < 0 and (len(self.getLegalActions(state)) > len(possibleAs))):
            return 0
        else:
            return bestASc

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestA=None
        bestASc = None
        if(len(self.getLegalActions(state))==0):
            return bestA
        elif(state not in self.qvals.keys()):
            return random.choice(self.getLegalActions(state))
        else:
            possibleAs = list(self.getLegalActions(state)) #TODO: Need to change this to randomly select from acitons unseen when all others are negative
            #print(possibleAs)
            for a in self.qvals[state].keys():
                qval = self.getQValue(state,a)
                #print(qval)
                if(bestASc is None or qval>=bestASc):
                    if(bestASc is not None and qval>bestASc):
                        possibleAs.remove(bestA)
                    bestASc = qval
                    bestA= a
                else:
                    possibleAs.remove(a)
            if(bestASc<0 and len(possibleAs)>1):
                possibleAs.remove(bestA)
                return random.choice(possibleAs)
            elif(len(possibleAs)>1 and bestASc == 0):
                return random.choice(possibleAs)
            else:
                return bestA

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        "*** YOUR CODE HERE ***"
        if(len(self.getLegalActions(state))==0):
            return None
        bestA=self.computeActionFromQValues(state)
        if(util.flipCoin(self.epsilon)):
            return random.choice(self.getLegalActions(state))
        else:
            return bestA

    def getMaxQ(self,nextState):
        #helper function to get maximum Qvalue of another state
        if(nextState not in self.qvals.keys()):
            return 0.0
        else:
            qvals=[]
            for a in self.qvals[nextState].keys():
                qvals.append(self.qvals[nextState][a][3])
        if(len(qvals)!=len(self.getLegalActions(nextState))):
            if(max(qvals)<0):
                return 0
            else:
                return max(qvals)
        return max(qvals)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # dict to get state to actions, and from each action, to its next states and the number of times we have done this,reward, and finally its qvalue
        # self.qvals[(1,1)]['north'] => [(1,2),10,50,32]
        # the thirs
        if(nextState in self.qvals.keys()):
            sample=reward + self.discount*self.getMaxQ(nextState)
        else:
            sample=reward


        if(state not in self.qvals.keys()):
            self.qvals[state]={}
            for a in self.getLegalActions(state):
                if(a!= action):
                    self.qvals[state][a] = [[],[0],[0],0]
                else:
                    self.qvals[state][action] = [[nextState],[1],[reward],self.alpha*sample]
        elif(state in self.qvals.keys()):
            if(action not in self.qvals[state].keys()): # we should never enter this function
                print('SHOULDNT BE HERE: state but no action')
                self.qvals[state][action] = [[nextState],[1],[reward],self.alpha*(sample)]
            else: # action and state already seen, in this new version, dont need to keep track of how often we have seen this value
                self.qvals[state][action][3] = (1-self.alpha)*self.getQValue(state, action) + self.alpha*sample
            # now we need to check if anyone else has this state as a q value
        '''
        older version, cleaning it up for less bugs maybe
        if(state not in self.qvals.keys()):
            if(nextState in self.qvals.keys()):
                for a in self.qvals[nextState].keys():
                    pqs = []
                    pqs.append(self.qvals[nextState][a][3])
                self.qvals[state]={action:[[nextState],[1],[reward],self.alpha*(reward+self.discount*max(pqs))]}
            else:
                self.qvals[state]={action:[[nextState],[1],[reward],self.alpha*reward]}
        elif(state in self.qvals.keys()):
            if(action not in self.qvals[state].keys()):
                #print('state but no action')
                if(nextState in self.qvals.keys()):
                    pqs = []
                    for a in self.qvals[nextState].keys():
                        pqs.append(self.qvals[nextState][a][3])
                    self.qvals[state]={action:[[nextState],[1],[reward],self.alpha*(reward+self.discount*max(pqs))]}
                else:
                    self.qvals[state]={action:[[nextState],[1],[reward],self.alpha*reward]}
            else: # action and state already seen, in this new version, dont need to keep track of how often we have seen this value
                maxqs = []
                if(nextState not in self.qvals.keys()):
                    newQ = reward
                    self.qvals[state][action][3] = (1-self.alpha)*self.qvals[state][action][3] + self.alpha*newQ
                else:
                    pqs = []
                    for a in self.qvals[nextState].keys():
                        pqs.append(self.qvals[nextState][a][3])
                    newQ = reward + self.discount*max(pqs)
                    self.qvals[state][action][3] = (1-self.alpha)*self.qvals[state][action][3] + self.alpha*newQ

        '''
        '''
        the below was my original qvalue iteration, but thats not what we are doing
        else: # action and state already seen
            print('are we even doing this')
            counter = 0
            tobeprobs=[]
            denom=0
            inList = False
            for nextStates in self.qvals[state][action][0]:
                if(nextStates == nextState):
                    inList=True
                    break
            if(inList == False):
                self.qvals[state][action][0].append(nextState)
                self.qvals[state][action][1].append(0) # this hsould work assuming the next for loop increments in appropriately
                self.qvals[state][action][2].append(reward)
            for nextStates in self.qvals[state][action][0]:
                if(nextStates == nextState): #we have seen this action to this state before, increment occurence counter
                    self.qvals[state][action][1][counter]+=1
                denom+=self.qvals[state][action][1][counter]
                tobeprobs.append(self.qvals[state][action][1][counter])
                counter+=1
            counter = 0
            while(counter<len(tobeprobs)):
                tobeprobs[counter] = tobeprobs[counter]/denom
                counter+=1
            counter = 0
            newQ=0
            for reward in self.qvals[state][action][2]:
                newQ+=tobeprobs[counter]*(reward+self.discount*self.getMaxQ(nextState))
                counter+=1
            self.qvals[state][action][3]=newQ
        '''
        '''
        elif(action not in self.qvals[state].keys()):
            self.qvals[state][action]=[nextState,1,reward,reward]

        elif(stuple not in self.qvals[stuple].keys()):
            self.qvals[stuple][nextState]=(1,reward) # we are assuming rewards are staying constant, they dont have to?
        else:
            numRecSoFar = self.qvals[stuple][nextState][0]
            self.qvals[stuple][nextState]=(numRecSoFar+1,reward)
        '''


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        #print(self.getWeights())
        for x in (self.featExtractor.getFeatures(state,action)):
        #    print(x)
            sum+= self.getWeights()[x]*self.featExtractor.getFeatures(state,action)[x]
        #print(sum)
        return sum
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        debug = False
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        isInside = False

        pqs = []
        for a in self.getLegalActions(nextState):
            #print(self.getQValue(nextState,a))
            pqs.append(self.getQValue(nextState,a))
        #print(action)
        if(len(pqs)==0):
            mpqv = 0
        else:
            mpqv = max(pqs)
        diffp1 = (reward + self.discount*mpqv)

        diff = diffp1 - self.getQValue(state,action)

        for x in (self.featExtractor.getFeatures(state,action)):
            self.getWeights()[x] = self.getWeights()[x] + self.alpha*diff*self.featExtractor.getFeatures(state,action)[x]



    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.getWeights())
            pass
