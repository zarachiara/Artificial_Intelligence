# perceptron_pacman.py
# --------------------
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


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data ):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for l in legalMoves:
                vectors[l] = self.weights * datum[l] #changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses


    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        self.features = trainingData[0][0]['Stop'].keys() # could be useful later
        
        # print("trainingdata")
        # print(trainingData)
        guessed = self.classify(trainingData)
        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            # Perceptron passing through the training data
            for i in range(len(trainingData)):
                "*** YOUR CODE HERE ***"
                # highScore = float("-inf")
                score = util.Counter()
                # goalY = float("-inf")
                moves = []
                for action in trainingData[i][1]: 
                    # print("action")
                    moves.append(action)
                #     print("action")
                #     print(action)
                # print("moves")
                # print(moves)
                currData = trainingData[i]
                # print("currData")
                # print(currData)
                # print("CurrData")
                # print(currData)
                # # print(currData)
                # print("self.legalLabels")
                # print(self.legalLabels)
                for action in moves:
                    # print("label")
                    # print(label)
                    score[action] = currData[0] * self.weights
                    # print("scoreLabel")
                    # print(score[action])
                    # print("self.weights.label")
                    # print(self.weights[label])
                    # print("self.weights")
                    # print(self.weights)
                    # print("all scores are here")
                    # print(score)
                    # for key in score.keys():
                    #     # print(score)
                    #     # print("score")
                    #     if score[key] > highScore:
                    #         highScore = score[key]
                    #         goalY = key
                    #         # print("goalY")
                    #         # print(goalY)
                goalY = score.argMax()
                realY = trainingLabels[i]
                # print("realY")
                # print(realY)
                # update weights
                if goalY != realY:
                    # print("type currData")
                    # print(type(currData))
                    # print("guessed")
                    # print(type(guessed[0]))
                    # print("guessed value")
                    # print(guessed)
                    self.weights = self.weights + currData[0]
                    self.weights = self.weights - trainingData[i][0][guessed[i]]
                    # print("self.weights")
                    # print(self.weights)
                    # print("self.weights type")
                    # print(type(self.weights))


