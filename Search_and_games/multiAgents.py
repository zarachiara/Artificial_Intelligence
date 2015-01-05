# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
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

import searchAgents
import sys

from random import randrange

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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


        "*** YOUR CODE HERE ***"
        # import pdb; pdb.set_trace()


        # IMPORT SEARCHAGENTS
        # IMPORT SYS

        # OLD GAME STATE INFO: oldFoodDist, oldFoodCount, oldGhostDist
        oldGhostDist = sys.maxsize
        oldFoodDist = sys.maxsize
        oldPacmanPos = currentGameState.getPacmanPosition()

        for oldGhost in currentGameState.getGhostStates():
          oldGhostPos = oldGhost.getPosition()
          oldGhostDist = min(oldGhostDist, util.manhattanDistance(oldGhostPos, oldPacmanPos))
          #oldGhostDist = min(oldGhostDist, searchAgents.mazeDistance(oldGhostPos, oldPacmanPos, currentGameState))
        for x, row in enumerate(currentGameState.getFood()):
          for y, value in enumerate(row):
            if value:
              oldFoodPos = (x, y)
              oldFoodDist = min(oldFoodDist, util.manhattanDistance(oldFoodPos, oldPacmanPos))
              #oldFoodDist = min(oldFoodDist, searchAgents.mazeDistance(oldFoodPos, oldPacmanPos, currentGameState))
        oldFoodCount = currentGameState.getFood().count()

        # NEW GAME STATE INFO: newFoodDist, newFoodCount, newGhostDist
        newGhostDist = sys.maxsize
        newFoodDist = sys.maxsize

        for newGhost in newGhostStates:
          newGhostPos = newGhost.getPosition()
          newGhostDist = min(newGhostDist, util.manhattanDistance(newGhostPos, newPos))
          #newGhostDist = min(newGhostDist, searchAgents.mazeDistance(newGhostPos, newPos, successorGameState))
        for x, row in enumerate(newFood):
          for y, value in enumerate(row):
            if value:
              newFoodPos = (x, y)
              newFoodDist = min(newFoodDist, util.manhattanDistance(newFoodPos, newPos))
              #newFoodDist = min(newFoodDist, searchAgents.mazeDistance(newFoodPos, newPos, successorGameState))
        newFoodCount = newFood.count()

        # I have: oldFoodDist, oldFoodCount, oldGhostDist
        #         newFoodDist, newFoodCount, newGhostDist

        # WIN/LOSE States
        if newGhostDist == 0:
          return -sys.maxsize
        if newFood.count() == 0:
          return sys.maxsize

        if newGhostDist == 1:
          return 0

        
        foodDistScalar = 1
        foodCountScalar = 1
        if newFoodDist < oldFoodDist:
          foodDistScalar = 2
        if newFoodCount < oldFoodCount:
          foodCountScalar = 500 # 5

        evaluation = (1.0/newFoodCount) * (1.0/newFoodDist) * foodDistScalar * foodCountScalar

        return int(evaluation*100000)
             

        # return successorGameState.getScore()

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
      to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

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
      Your minimax agent (question 7)
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
        """
        "*** YOUR CODE HERE ***"

        actions = gameState.getLegalActions(0)
        best_score = float('-inf')
        best_action = actions[0]

        for action in actions:
          clone = gameState.generateSuccessor(0, action)
          score = self.minPlay(clone, 1, gameState.getNumAgents(), self.depth)
          if score > best_score:
            best_action = action
            best_score = score

        return best_action

    def minPlay(self, gameState, agent, numAgents, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent)

        best_score = float('inf')

        for action in actions:
          clone = gameState.generateSuccessor(agent, action)
          if agent == numAgents-1:
            score = self.maxPlay(clone, 0, numAgents, depth-1)
          else:
            score = self.minPlay(clone, agent+1, numAgents, depth)
          if score < best_score:
            best_action = action
            best_score = score
        return best_score

    def maxPlay(self, gameState, agent, numAgents, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent)

        best_score = float('-inf')

        for action in actions:
          clone = gameState.generateSuccessor(agent, action)
          score = self.minPlay(clone, agent+1, numAgents, depth)
          if score > best_score:
            best_action = action
            best_score = score
        return best_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        # FROM RANDOM IMPORT RANDRANGE
        actions = gameState.getLegalActions(0)
        best_score = float('-inf')
        best_action = actions[0]

        for action in actions:
          clone = gameState.generateSuccessor(0, action)
          score = self.randPlay(clone, 1, gameState.getNumAgents(), self.depth)
          if score > best_score:
            best_action = action
            best_score = score

        return best_action

    def randPlay(self, gameState, agent, numAgents, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent)

        expectation = 0.0

        for action in actions:
          clone = gameState.generateSuccessor(agent, action)
          if agent == numAgents-1:
            score = self.maxPlay(clone, 0, numAgents, depth-1)
          else:
            score = self.randPlay(clone, agent+1, numAgents, depth)
          expectation += score

        return expectation / len(actions)

    def maxPlay(self, gameState, agent, numAgents, depth):
        if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent)

        best_score = float('-inf')

        for action in actions:
          clone = gameState.generateSuccessor(agent, action)
          score = self.randPlay(clone, agent+1, numAgents, depth)
          if score > best_score:
            best_action = action
            best_score = score
        return best_score

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 9).

      DESCRIPTION:
        - Winning state returns huge positive value
        - Losing state returns huge negative value
        - I used three game state metrics:
          - foodCount: number of food pieces on the board
          - ghostDist: distance from Pacman to the nearest ghost
          - foodDist: distance from Pacman to the nearest food
        - My evaluation normalizes these values (in case they return wild values),
          multiplies each value by a scalar (depending on how important they are),
          and then sums everything up to return a single value.
    """
    "*** YOUR CODE HERE ***"

    # IMPORT SEARCHAGENTS

    ghostDist = 10000#sys.maxsize
    foodDist = 10000#sys.maxsize
    pacmanPos = currentGameState.getPacmanPosition()

    for ghost in currentGameState.getGhostStates():
      ghostPos = ghost.getPosition()
      ghostDist = min(ghostDist, util.manhattanDistance(ghostPos, pacmanPos))
      #ghostDist = min(ghostDist, searchAgents.mazeDistance(ghostPos, pacmanPos, currentGameState))

    
    for x, row in enumerate(currentGameState.getFood()):
      for y, value in enumerate(row):
        if value:
          foodPos = (x, y)
          foodDist = min(foodDist, util.manhattanDistance(foodPos, pacmanPos))
          #foodDist = min(foodDist, searchAgents.mazeDistance(foodPos, pacmanPos, currentGameState))
    
    foodCount = currentGameState.getNumFood()
    gameScore = currentGameState.getScore()
    capsuleCount = len(currentGameState.getCapsules())

    # Use: foodCount, ghostDist, foodDist, gameScore

    if currentGameState.isWin() or foodCount == 0:
      # print "Win State Achieved!"
      # print "\n"
      return sys.maxsize
    if currentGameState.isLose():
      # print "Lose State Achieved :("
      # print "\n"
      return -sys.maxsize - 1

    evaluation = 10000.0*gameScore - 200.0*foodCount - 5.0*foodDist - 1000.0*capsuleCount#+ float(ghostDist)
    # print "Gamescore: " + str(gameScore)
    # print "Food Count: " + str(foodCount)
    # print "Food Distance: " + str(foodDist)
    # print "Ghost Distance: " + str(ghostDist)
    # print "Evaluation: " + str(evaluation)
    # print "\n"
    return evaluation
#    return (1000000*gameScore) + (100000.0/foodCount) + (1000.0/foodDist) + (ghostDist)

    # return sys.maxsize - foodCount*1000 - foodDist*10 + ghostDist


    """
    vector = [foodCount, foodDist, ghostDist]
    normalizedVector = util.normalize(vector)

    foodCountScalar = 10000000
    foodDistScalar = 5
    ghostDistScalar = -2
    scalarsVector = [foodCountScalar, foodDistScalar, ghostDistScalar]

    return sum([a*b for a,b in zip(normalizedVector, scalarsVector)])
    """


# Abbreviation
better = betterEvaluationFunction

