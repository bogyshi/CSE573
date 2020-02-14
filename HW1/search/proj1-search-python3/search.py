# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from game import Directions
    #from sets import Set
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    dmap = {'North':n,'South':s,'West':w,'East':e}
    ST = problem.getStartState()
    nodes = util.Stack()
    nodes.push(ST)
    alreadyDiscovered=set()
    path = []
    nodePath=[]
    parentMap={}
    parentDMap = {}
    firstTime=True
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    while(not nodes.isEmpty()):
        tempNode = nodes.pop()
        if(not firstTime):
            loc = tempNode[0]
        else:
            loc = tempNode
            firstTime = False
        nodePath.append(loc)
        #print(alreadyDiscovered)
        if(problem.isGoalState(loc)):
            '''
            need to add traversal code
            '''
    #        print('done at ' + str(loc))
            loc2=loc
            path.append(parentDMap[loc2])
            while(parentMap[loc2]!=ST):
                loc2= parentMap[loc2]
                path.append(parentDMap[loc2])
            break
            #return []
        elif(loc in alreadyDiscovered):
    #        print('seen this node: ' + str(tempNode))
            pass
            #print(tempNode)
        else:
            alreadyDiscovered.add(loc)
            for x in problem.getSuccessors(loc):
                if(x[0] in parentMap.values()):
                    pass
                else:
                    parentMap[x[0]] = loc
                    parentDMap[x[0]] = x[1]
                nodes.push(x)
    path.reverse()
    #print(path)
    return path

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    #from sets import Set
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    ST = problem.getStartState()
    nodes = util.Queue()
    nodes.push(ST)
    alreadyDiscovered=set()
    path = []
    nodePath=[]
    parentMap={}
    parentDMap = {}
    firstTime=True
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    while(not nodes.isEmpty()):
        tempNode = nodes.pop()
        if(not firstTime):
            loc = tempNode[0]
        else:
            loc = tempNode
            firstTime = False
        nodePath.append(loc)
        #print(alreadyDiscovered)
        if(problem.isGoalState(loc)):
    #        print('done at ' + str(loc))
            loc2=loc
            path.append(parentDMap[loc2])
            while(parentMap[loc2]!=ST):
                loc2= parentMap[loc2]
                path.append(parentDMap[loc2])
            break
            #return []
        elif(loc in alreadyDiscovered):
    #        print('seen this node: ' + str(tempNode))
            pass
            #print(tempNode)
        else:
            alreadyDiscovered.add(loc)
            for x in problem.getSuccessors(loc):
                if(x[0] in parentMap.values() or x[0] in parentMap.keys()):
                    pass
                else:
                    parentMap[x[0]] = loc
                    parentDMap[x[0]] = x[1]
                nodes.push(x)
    path.reverse()
    #print(path)
    return path
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    #from sets import Set
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    ST = problem.getStartState()
    nodes = util.PriorityQueue()
    nodes.push(ST,0)
    cost = 0
    alreadyDiscovered=set()
    path = []
    nodePath=[]
    parentMap={}
    parentDMap = {}
    parentCMap={ST:0}
    firstTime=True
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Is this other place a goal?", problem.isGoalState((33,14)))

    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    while(not nodes.isEmpty()):
        tempNode = nodes.pop()
        if(not firstTime):
            loc = tempNode[0]
            cost = tempNode[2]
        else:
            loc = tempNode
            firstTime = False
        #print(cost)
        nodePath.append(loc)
        #print(alreadyDiscovered)
        if(problem.isGoalState(loc)):
            '''
            need to add traversal code
            '''
            #print('done at ' + str(loc))
            loc2=loc
            path.append(parentDMap[loc2])
            while(parentMap[loc2]!=ST):
                loc2= parentMap[loc2]
                path.append(parentDMap[loc2])
            break
            #return []
        elif(loc in alreadyDiscovered):
            #print('seen this node: ' + str(tempNode))
            pass
            #print(tempNode)
        else:
            alreadyDiscovered.add(loc)
            for x in problem.getSuccessors(loc):
                cost = x[2]
                #if(x[0] in parentMap.values() or x[0] in parentMap.keys()):
                #    pass
                #else:
                if(x[0] not in parentMap.keys()):
                    parentMap[x[0]] = loc
                    parentDMap[x[0]] = x[1]
                    parentCMap[x[0]] = parentCMap[loc]+cost
                elif(parentCMap[loc]+cost<parentCMap[x[0]]):
                    parentMap[x[0]] = loc
                    parentDMap[x[0]] = x[1]
                    parentCMap[x[0]] = parentCMap[loc]+cost

                nodes.push(x,parentCMap[x[0]])
    path.reverse()
    #print(path)
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from game import Directions
    #from sets import Set
    s = Directions.SOUTH
    w = Directions.WEST
    e = Directions.EAST
    n = Directions.NORTH
    ST = problem.getStartState()
    nodes = util.PriorityQueue()
    nodes.push(ST,0)
    cost = 0
    alreadyDiscovered=set()
    path = []
    nodePath=[]
    parentMap={}
    parentDMap = {}
    parentCMap={ST:0}
    firstTime=True
    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Is this other place a goal?", problem.isGoalState((33,14)))

    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    while(not nodes.isEmpty()):
        tempNode = nodes.pop()
        if(not firstTime):
            loc = tempNode[0]
            cost = tempNode[2]
            hcost = heuristic(loc,problem)
        else:
            loc = tempNode
            firstTime = False
            hcost = heuristic(loc,problem)
        nodePath.append(loc)
        #print(alreadyDiscovered)
        if(problem.isGoalState(loc)):
            '''
            need to add traversal code
            '''
            #print('done at ' + str(loc))
            loc2=loc
            path.append(parentDMap[loc2])
            while(parentMap[loc2]!=ST):
                loc2= parentMap[loc2]
                path.append(parentDMap[loc2])
            break
            #return []
        elif(loc in alreadyDiscovered):
            #print('seen this node: ' + str(tempNode))
            pass
            #print(tempNode)
        else:
            alreadyDiscovered.add(loc)
            for x in problem.getSuccessors(loc):
                cost = x[2]
                #print(cost)

                #if(x[0] in parentMap.values() or x[0] in parentMap.keys()):
                #    pass
                #else:
                if(x[0] not in parentMap.keys()):
                    parentMap[x[0]] = loc
                    parentDMap[x[0]] = x[1]
                    parentCMap[x[0]] = parentCMap[loc]+cost
                elif(parentCMap[loc]+cost<parentCMap[x[0]]):
                    parentMap[x[0]] = loc
                    parentDMap[x[0]] = x[1]
                    parentCMap[x[0]] = parentCMap[loc]+cost
                #print(heuristic(x[0],problem))
                nodes.push(x,parentCMap[x[0]]+heuristic(x[0],problem))
    path.reverse()
    #print(path)
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
