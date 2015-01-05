# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy

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

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    queue = util.Queue()
    queue.push((start, []))
    mySet = set(start)

    while queue:
        state, actions = queue.pop()
        if problem.isGoalState(state):
            return actions
        for state, action, _ in problem.getSuccessors(state):
            if state not in mySet:
                mySet.add(state)
                queue.push((state, actions+[action]))
    return False

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth.

    Begin with a depth of 1 and increment depth by 1 at every step.
    """
    "*** YOUR CODE HERE ***"

    total_depth = 1

    while True:

        frontier = [(problem.getStartState(), [], 0)] #coord, actions, depth
        explored = []
        depth = 0

        while frontier:

            if not frontier:
                return False

            coord, actions, depth = frontier.pop()
            explored.append(coord)
            if problem.isGoalState(coord):
                return actions
            if depth < total_depth:
                for child in problem.getSuccessors(coord):
                    #pdb.set_trace()
                    new_coord = child[0]
                    new_action = child[1]
                    #action = child[1]
                    if not [node for node in frontier if node[0] == new_coord] and new_coord not in explored:
                        frontier.append((new_coord, actions + [new_action], depth+1))

        total_depth += 1

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    # initialize frontier and put starting node on the frontier
    frontier = util.PriorityQueue()
    start = problem.getStartState(), None, 0

    #print start
    frontier.push([start], 0)

    # print "problem"
    # print problem

    # initialized visited list 
    visited = []

    # initialize return list, which is a list of actions 
    actions = []

    #initialize path cost list Key is the path, and value is the action list 
    paths = {}
    
    while not frontier.isEmpty():
        # print "frontier"
        # print frontier
        # pop the first element in priority queue (based on f_values)# from the frontier and push it to explored and ret
        
        path = frontier.pop()
        # print "path"
        # print path 
        # consist of the action lists 
        explore = path[-1] #grab last element of path list, the action lists
        # print "path"
        # print path
        if explore[0] in visited:
            continue
        # initialize g value
        g = 0

        # Case where the problem is the goal
        if problem.isGoalState(explore[0]):
            actions = []
            for state, action, cost in path:
                if action != None:
                    actions.append(action)
            return actions

        for state, action, cost in path:  # calculate g every time NOT EFFICIENT
            g = g + cost

        # print "g value for node: "
        # print explore[0]
        # print g

        # inititalize the coordinates (used as argument 
        # for successors)
        coords = explore[0]

        # TODO check if nodes have already been visited
        visited.append(coords)
        # print "visited: "
        # print visited

        # call successors of explore
        successors = problem.getSuccessors(coords)
 
        # push the successors in frontier with min f_values as the priority
        for child in successors: 
     
            # keep track of state of the child 
            childCoords = child[0]

            # check if state was already in visited
            if childCoords in visited:
                continue

            # takes the list of path 
            temp = list(path)
            # print "appending child"
            # print child
            # print "with g value"
            # print g
            # appends it to the child 
            temp.append(child)
       #     frontier.push(path,(paths[coords + childCoords] + heuristic(coords, problem)))
            # print 'pushing path: '
            # print temp
            frontier.push(temp, g + child[2] + heuristic(childCoords, problem))

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
