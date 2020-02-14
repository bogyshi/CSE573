# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import pdb

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


    def customEvalHelpFunction(self,pos,food,ghosts):
        #python pacman.py --frameTime 0 -p ReflexAgent -k 1 -q -n 20
        w=food.width
        h=food.height
        x=0
        y=0
        dists=[]
        gdist=[]
        posScore=0
        negScore=0
        while(x<w):
            y=0
            while(y<h):
                if(food[x][y]):
                    dists.append(manhattanDistance(pos,(x,y)))
                y+=1
            x+=1
        for g in ghosts:
            gdist.append(manhattanDistance(pos,g.configuration.pos))
        if(len(dists)==0):
            return 1000000
        if(min(dists)<2):
            posScore = 10
        else:
            posScore = 1/(min(dists))
        if(min(gdist)<3):
            negScore = 100
        else:
            negScore = 1/min(gdist)
        #print(pos)
        #print(type(food))
        #print(str(ghosts))
        return ( posScore - negScore)

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        custScore = self.customEvalHelpFunction(newPos,newFood,newGhostStates)
        return successorGameState.getScore()+custScore

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def recurseHelper(self,aCounter,d,md,GS):
        if(d==md or GS.isWin() or GS.isLose()):
            return(self.evaluationFunction(GS),None)# this might be wrong, unsure if we need to pass around actions or not
        else:
            NA = GS.getNumAgents()
            if(aCounter>0):
                if(aCounter == NA-1):
                    d+=1
                mScore = 999999
                mAction = None
                for f in GS.getLegalActions(aCounter):
                    tempScore,whocares = self.recurseHelper((aCounter+1)%NA,d,md,GS.generateSuccessor(aCounter,f))
                    if(tempScore<mScore):
                        mScore = tempScore
                        mAction = f
                return mScore,mAction
            else:
                mScore=-99999
                mAction = None
                for f in GS.getLegalActions(aCounter):
                    tempScore,whocares = self.recurseHelper((aCounter+1)%NA,d,md,GS.generateSuccessor(aCounter,f))
                    if(tempScore>mScore):
                        mScore = tempScore
                        mAction = f
                return mScore,mAction

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        sIndex = (self.index) # appears to always be pacman, or zero
        numAgents = gameState.getNumAgents()
        dCounter = 0
        maxD = self.depth
        #print(maxD) # this means no need for recursion like i thought it might
        # nevermind it totally does
        aCounter = sIndex
        score,move=(self.recurseHelper(sIndex,0,maxD,gameState))
        #util.raiseNotDefined()
        return move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def recurseHelper(self,aCounter,d,md,GS,alpha,beta):
        if(d==md or GS.isWin() or GS.isLose()):
            val = self.evaluationFunction(GS)
            #print(val)
            return val,None,0,0# this might be wrong, unsure if we need to pass around actions or not
        else:
            NA = GS.getNumAgents()
            if(aCounter>0):
                if(aCounter == NA-1):
                    d+=1
                #print('Ghost num: '+ str(aCounter))

                mScore = 999999
                mAction = None
                for f in GS.getLegalActions(aCounter):
                #    if(d==2):
                #        print('GhostAction at depth: '+str(d))
                #        print(alpha)
                #        print(beta)
                    tempScore,whocares,pAlpha,pBeta = self.recurseHelper((aCounter+1)%NA,d,md,GS.generateSuccessor(aCounter,f),alpha,beta)
                    if(tempScore<mScore):
                        mScore = tempScore
                        mAction = f
                    if(alpha is None or alpha > mScore):
                        alpha = mScore
                    if(beta is None):
                        pass
                    else:
                        if(tempScore < beta): # if the beta i got from a higher depth pacman (only place where its updated) says val X, dont investiage anything lower
                            break
                return mScore,mAction,alpha,beta
            else:
                mScore=-99999
                #print('Pacman')
                mAction = None
                for f in GS.getLegalActions(aCounter):
                    #if(d==1):
                    #    print('PacmanAction at depth: ' + str(d))
                    #    print(alpha)
                    #    print(beta)
                    tempScore,whocares,pAlpha,pBeta = self.recurseHelper((aCounter+1)%NA,d,md,GS.generateSuccessor(aCounter,f),alpha,beta)
                    if(tempScore>mScore):
                        mScore = tempScore
                        mAction = f
                    if(beta is None or beta<mScore):
                        beta = mScore
                    if(alpha is not None):
                        if(tempScore>alpha):
                            break # if the alpha from a higher ghost is lower than something i would maximize, dont bother looking
                return mScore,mAction,alpha,beta

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        sIndex = (self.index) # appears to always be pacman, or zero
        numAgents = gameState.getNumAgents()
        dCounter = 0
        maxD = self.depth
        #print(maxD) # this means no need for recursion like i thought it might
        # nevermind it totally does
        aCounter = sIndex
        score,move,alpha,beta=(self.recurseHelper(sIndex,0,maxD,gameState,None,None))
        #util.raiseNotDefined()
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def recurseHelper(self,aCounter,d,md,GS):
        if(d==md or GS.isWin() or GS.isLose()):
            val = self.evaluationFunction(GS)
            return val,None# this might be wrong, unsure if we need to pass around actions or not
        else:
            NA = GS.getNumAgents()
            if(aCounter>0):
                #print('Ghost num: '+ str(aCounter))
                if(aCounter == NA-1):
                    d+=1
                mScore = 0
                mAction = None
                nActions=0
                for f in GS.getLegalActions(aCounter):
                    tempScore,whocares = self.recurseHelper((aCounter+1)%NA,d,md,GS.generateSuccessor(aCounter,f))
                    mScore+=tempScore
                    nActions+=1
                mScore = mScore/nActions
                return mScore,None #action of ghost shouldnt matter
            else:
                #print('Pacman')
                mScore=-99999
                mAction = None
                for f in GS.getLegalActions(aCounter):
                    #print('PacmanAction at depth: ' + str(d))
                    tempScore,whocares = self.recurseHelper((aCounter+1)%NA,d,md,GS.generateSuccessor(aCounter,f))
                    if(tempScore>mScore):
                        mScore = tempScore
                        mAction = f
                return mScore,mAction

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        sIndex = (self.index) # appears to always be pacman, or zero
        numAgents = gameState.getNumAgents()
        dCounter = 0
        maxD = self.depth
        #print(maxD) # this means no need for recursion like i thought it might
        # nevermind it totally does
        aCounter = sIndex
        score,move=(self.recurseHelper(sIndex,0,maxD,gameState))
        #util.raiseNotDefined()
        return move


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    Tried some manual weighting on various features such as distance to food, ghosts, and if a ghost was scared. Should add some code
    for encouraging the eating of super capsules and then ghosts
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghosts]


    w=food.width
    h=food.height
    x=0
    y=0
    dists=[]
    gdist=[]
    posScore=0
    negScore=0
    #print('start')
    #print(pos)
    while(x<w):
        y=0
        while(y<h):
            if(food[x][y]):
                dists.append(manhattanDistance(pos,(x,y)))
            y+=1
        x+=1
    for g in ghosts:
        gdist.append(manhattanDistance(pos,g.configuration.pos))
    if(len(dists)==0):
        return 1010101010
    elif(min(dists)<2):
        #print('we close!')
        posScore += 5
    elif(min(dists)<3):
        #print('sorta close')
        posScore = 0.5
    else:
        #print('not close    ')
        posScore = 1/(min(dists))
    gCounter = 0
    while(gCounter<len(ghosts)):
        if(gdist[gCounter] < 3 and scaredTimes[gCounter]==0):
            negScore +=3
        elif(gdist[gCounter] >= 3 and scaredTimes[gCounter]>0):
            negScore += -1
        elif(gdist[gCounter] < 3 and scaredTimes[gCounter]>0):
            negScore+= -2
        elif( scaredTimes[gCounter]>0):
            negScore += -1 * (1/gdist[gCounter])
        else:
            negScore += 1/gdist[gCounter]

        gCounter+=1
    #negScore=0
    #negScore += len(dists)
    #if(min(gdist)<3):
    #    negScore = 100
    #else:
    #    negScore = 1/min(gdist)
    #print(pos)
    #print(type(food))
    #print(str(ghosts))
    rScore =  posScore -negScore + currentGameState.getScore()
    #print(rScore)
    #print('end')
    return (rScore)

# Abbreviation
better = betterEvaluationFunction
