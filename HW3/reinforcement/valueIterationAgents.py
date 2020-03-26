# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        i = 0
        for s in (self.mdp.getStates()):
            self.values[s]=0
            if(self.mdp.isTerminal(s)):
                pass
            else:
                pass
                #print(self.mdp.getTransitionStatesAndProbs(s,self.mdp.getPossibleActions(s)[0]))
        while(i<self.iterations):
            toUpdate = self.values.copy()
            for s in self.mdp.getStates():
                possibleAs = []
                for a in self.mdp.getPossibleActions(s):
                    totAVal = 0
                    for news,prob in self.mdp.getTransitionStatesAndProbs(s,a):
                        totAVal +=prob*(self.mdp.getReward(s,a,news)+self.discount*self.values[news])
                    possibleAs.append(totAVal)
                if(len(possibleAs)>0):
                    toUpdate[s] = max(possibleAs)
            self.values = toUpdate
            #print(self.values)
            i+=1


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        qval = 0
        for news,prob in self.mdp.getTransitionStatesAndProbs(state,action):
            qval += prob*(self.mdp.getReward(state,action,news)+self.discount*self.values[news])
        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestA= None
        bestAScore = None
        if(self.mdp.isTerminal(state)):
            return None
        else:
            for a in self.mdp.getPossibleActions(state):
                totAVal = 0
                for news,prob in self.mdp.getTransitionStatesAndProbs(state,a):
                    totAVal += prob*(self.mdp.getReward(state,a,news)+self.discount*self.values[news])
                if(bestAScore is None or bestAScore<totAVal):
                    bestAScore = totAVal
                    bestA = a
        return bestA

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        i = 0
        for s in (self.mdp.getStates()):
            self.values[s]=0
            if(self.mdp.isTerminal(s)):
                pass
            else:
                pass
                #print(self.mdp.getTransitionStatesAndProbs(s,self.mdp.getPossibleActions(s)[0]))
        while(i < self.iterations):
            #toUpdate = self.values.copy()
            for s in self.mdp.getStates():
                if(i >= self.iterations):
                    break
                elif(self.mdp.isTerminal(s)):
                    pass
                else:
                    possibleAs = []
                    for a in self.mdp.getPossibleActions(s):
                        totAVal = 0
                        for news,prob in self.mdp.getTransitionStatesAndProbs(s,a):
                            totAVal +=prob*(self.mdp.getReward(s,a,news)+self.discount*self.values[news])
                        possibleAs.append(totAVal)
                    if(len(possibleAs)>0):
                        self.values[s] = max(possibleAs)
                i+=1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        i = 0
        # first get all predecessors
        predDict = {}
        for s in (self.mdp.getStates()):
            predDict[s] = set()
        for s in (self.mdp.getStates()):
            self.values[s]=0
            for s in self.mdp.getStates():
                if(self.mdp.isTerminal(s)):
                    pass # terminals cant take actions to get to the state, so not a predecessor
                else:
                    for a in self.mdp.getPossibleActions(s):
                        for news,prob in self.mdp.getTransitionStatesAndProbs(s,a):
                            if(prob>0):
                                predDict[news].add(s)
        #now we have our predecessors, now we just need to do this priority queue stuff
        pqueue = util.PriorityQueue()
        for s in predDict.keys():
            if(self.mdp.isTerminal(s)):
                pass
            else:
                todiff = self.values[0]
                qs = []
                for a in self.mdp.getPossibleActions(s):
                    qs.append(self.computeQValueFromValues(s, a))
                prior = -1*abs(todiff-max(qs))
                pqueue.push(s,prior)

        # initialized priority queue

                #print(self.mdp.getTransitionStatesAndProbs(s,self.mdp.getPossibleActions(s)[0]))
        while(i < self.iterations and (not pqueue.isEmpty())):
            #toUpdate = self.values.copy()
            s = pqueue.pop()
            if(i >= self.iterations):
                break
            elif(self.mdp.isTerminal(s)):
                pass
            else:
                # the below may be some unecessary computation, but we can worry about that later
                possibleAs = []
                for a in self.mdp.getPossibleActions(s):
                    totAVal = 0
                    for news,prob in self.mdp.getTransitionStatesAndProbs(s,a):
                        totAVal +=prob*(self.mdp.getReward(s,a,news)+self.discount*self.values[news])
                    possibleAs.append(totAVal)
                if(len(possibleAs)>0):
                    self.values[s] = max(possibleAs)
                for pred in predDict[s]:
                    prevPVal = self.values[pred]
                    qs=[]
                    for a in self.mdp.getPossibleActions(pred):
                        qs.append(self.computeQValueFromValues(pred, a))
                    prior = abs(prevPVal-max(qs))
                    if(prior>self.theta):
                        pqueue.update(pred,-1*prior)
                i+=1
