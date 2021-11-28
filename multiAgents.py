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

        newFoodDistances = [manhattanDistance(newPos, pos) for pos in newFood.asList()]
        newCapsuleDistances = [manhattanDistance(newPos, pos2) for pos2 in successorGameState.getCapsules()]
        newGhostDistancesToPacman = list()

        minGhostDistance = 0
        minFoodDistance = 0
        minCapsuleDistance = 0
        avGhostDistance = 0

        score = successorGameState.getScore()

        result = 0

        if len(newGhostStates) != 0:
            for state in newGhostStates:
                newGhostDistancesToPacman.append(manhattanDistance(newPos, state.getPosition()))

        if len(newGhostDistancesToPacman) != 0:
            minGhostDistance = min(newGhostDistancesToPacman)

        if len(newFoodDistances) != 0:
            minFoodDistance = min(newFoodDistances)

        if len(newCapsuleDistances) != 0:
            minCapsuleDistance = min(newCapsuleDistances)

        result += minGhostDistance
        result -= minFoodDistance
        result += score

        return result

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

        def minimax(state, agent, depth):

            if state.isWin() or state.isLose() or depth == self.depth:
               return  (self.evaluationFunction(state), None)

            elif agent == 0:
                return maxState(state, agent, depth)

            elif agent != 0:
                return minState(state, agent, depth)

        def maxState(state, agent, depth):
            compare = float("-inf")
            v = float("-inf")
            resultAction = None
            legalActionsForPacman = state.getLegalActions(agent)
            for action in legalActionsForPacman:
                compare = minimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth)[0]
                if compare > v:
                    v = compare
                    resultAction = action

            return (v, resultAction)

        def minState(state, agent, depth):
            compare = float("inf")
            v = float("inf")
            resultAction = None
            legalActionsForGhost = state.getLegalActions(agent)
            for action in legalActionsForGhost:
                if (agent + 1) != state.getNumAgents():
                    compare = minimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth)[0]
                else:
                    compare = minimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth + 1)[0]
                if compare < v:
                    v = compare
                    resultAction = action

            return (v, resultAction)

        return maxState(gameState,0,0)[1]
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(state, agent, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
               return  (self.evaluationFunction(state), None)

            elif agent == 0:
                return maxState(state, agent, depth, alpha, beta)

            elif agent != 0:
                return minState(state, agent, depth, alpha, beta)

        def maxState(state, agent, depth, alpha, beta):
            compare = float("-inf")
            v = float("-inf")
            resultAction = None
            legalActionsForPacman = state.getLegalActions(agent)
            for action in legalActionsForPacman:
                compare = alphaBeta(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth, alpha, beta)[0]

                if compare > v:
                    v = compare
                    resultAction = action

                if v > beta:
                    return (v, resultAction)

                alpha = max(alpha, v)
            return (v, resultAction)

        def minState(state, agent, depth, alpha, beta):
            compare = float("inf")
            v = float("inf")
            resultAction = None
            legalActionsForGhost = state.getLegalActions(agent)
            for action in legalActionsForGhost:
                if (agent + 1) != state.getNumAgents():
                    compare = alphaBeta(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth, alpha, beta)[0]
                else:
                    compare = alphaBeta(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth + 1, alpha, beta)[0]

                if compare < v:
                    v = compare
                    resultAction = action

                if v < alpha:
                    return (v, resultAction)

                beta = min(beta, v)

            return (v, resultAction)

        return maxState(gameState, 0, 0, float("-inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
               return  (self.evaluationFunction(state), None)

            elif agent == 0:
                return maxState(state, agent, depth)

            elif agent != 0:
                return minState(state, agent, depth)

        def maxState(state, agent, depth):
            compare = float("-inf")
            v = float("-inf")
            resultAction = None
            legalActionsForPacman = state.getLegalActions(agent)
            for action in legalActionsForPacman:
                compare = expectimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth)[0]
                if compare > v:
                    v = compare
                    resultAction = action

            return (v, resultAction)

        def minState(state, agent, depth):
            dummy = 0
            v = list()
            resultAction = None
            legalActionsForGhost = state.getLegalActions(agent)
            for action in legalActionsForGhost:
                if (agent + 1) != state.getNumAgents():
                    dummy = expectimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth)[0]
                else:
                    dummy = expectimax(state.generateSuccessor(agent, action), (agent + 1) % state.getNumAgents(), depth + 1)[0]
                v.append(dummy)
                resultAction = action

            return (sum(v)/len(v), resultAction)

        return maxState(gameState, 0,0)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    score = currentGameState.getScore()
    foodNum = len(food.asList())
    foodDistances = [manhattanDistance(pos, pos2) for pos2 in food.asList()]
    capsuleDistances = [manhattanDistance(pos, pos2) for pos2 in currentGameState.getCapsules()]
    ghostDistancesToPacman = list()

    minGhostDistance = 0
    minFoodDistance = 0
    minCapsuleDistance = 0

    result = 0

    if len(ghostStates) != 0:
        for state in ghostStates:
            ghostDistancesToPacman.append(manhattanDistance(pos, state.getPosition()))

    if len(ghostDistancesToPacman) != 0:
        minGhostDistance = min(ghostDistancesToPacman)

    if len(foodDistances) != 0:
        minFoodDistance = min(foodDistances)

    if len(capsuleDistances) != 0:
        minCapsuleDistance = min(capsuleDistances)

    result += minGhostDistance
    result -= minFoodDistance
    result -= minCapsuleDistance*2
    result += score


    comparator = 0
    count = 0
    position = 0
    if len(scaredTimes) != 0:
        for time in scaredTimes:
            if time > 0 and time > comparator:
                comparator = time
                position = count
            count += 1
        result -= manhattanDistance(pos, ghostStates[position].getPosition()) * 2

    return result


# Abbreviation
better = betterEvaluationFunction
