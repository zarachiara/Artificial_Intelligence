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
import logic
from sets import Set

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

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostSearchProblem)
        """
        util.raiseNotDefined()

    def terminalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionSearchProblem
        """
        util.raiseNotDefined()

    def result(self, state, action):
        """
        Given a state and an action, returns resulting state and step cost, which is
        the incremental cost of moving to that successor.
        Returns (next_state, cost)
        """
        util.raiseNotDefined()

    def actions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

    def getWidth(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        util.raiseNotDefined()

    def getHeight(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        util.raiseNotDefined()

    def isWall(self, position):
        """
        Return true if position (x,y) is a wall. Returns false otherwise.
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


def atLeastOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at least one of the expressions in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> #print logic.pl_true(atleast1,model1)
    False
    >>> model2 = {A:False, B:True}
    >>> #print logic.pl_true(atleast1,model2)
    True
    >>> model3 = {A:True, B:True}
    >>> #print logic.pl_true(atleast1,model2)
    True
    """
    "*** YOUR CODE HERE ***"
    first = expressions[0]
    for expression in expressions[1::]:
        first = logic.Expr("|", first, expression)
    return first

def atMostOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that at most one of the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    if len(expressions) == 1:
        #print("ONLY ONE")
        return logic.Expr("|", expressions[0], logic.Expr("~", expressions[0]))
    """
    clauses = []
    for expression in expressions:
        str_ret = []
        for expr in expressions:
            if expression != expr:
                str_ret.append(logic.Expr("~", expr))
        str_ret = logic.associate("&", str_ret)
        expr = logic.Expr(">>", expression, str_ret)
        clauses.append(expr)
    result = logic.associate("&", clauses)
    ##print(result)
    return result
    """
    clauses = []
    i = 0
    for expression in expressions:
        for expr in expressions[i::]:
            if expression != expr:
                thing = logic.Expr("|", logic.Expr("~", expression), logic.Expr("~", expr))
                clauses.append(thing)
        i += 1
    result = logic.associate("&", clauses)
    ##print(result)
    return result
    #"""


def exactlyOne(expressions) :
    """
    Given a list of logic.Expr instances, return a single logic.Expr instance in CNF (conjunctive normal form)
    that represents the logic that exactly one of the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    return logic.Expr("&", atLeastOne(expressions), atMostOne(expressions))


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[3]":True, "P[3,4,1]":True, "P[3,3,1]":False, "West[1]":True, "GhostScary":True, "West[3]":False, "South[2]":True, "East[1]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> #print plan
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    # plan consists of all the actions that are true.  These are no in order yet. 
    plan = []
    for key in model.keys():
        symbols = logic.PropSymbolExpr.parseExpr(key)
        symbol_string = symbols[0]
        if symbol_string in actions: 
            if (model[key] == True):
                plan.append(key)
    # second for loop list actions in order based on time 
    ordered_plan = []
    counter = 0
    while counter < len(plan):
        for actions in plan:
            parse = logic.PropSymbolExpr.parseExpr(actions)
            direction = parse[0]
            time = int(parse[1])
            if (time == counter):
                ordered_plan.append(direction)
        counter = counter + 1
    return ordered_plan


def positionLogicPlan(problem):
    """
    Given an instance of a PositionSearchProblem, return a list of actions that lead to the goal.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    kb = logic.PropKB() # add axioms here
    t_max = 50
    start = problem.getStartState()

    # intial state; only 1 Pacman at intial time
    intial = logic.PropSymbolExpr("P", start[0], start[1], 0)
    kb.tell(intial)
    for h in range(1, problem.getHeight() + 1):
        for w in range(1, problem.getWidth() + 1):
            if (w,h) != start: # no contradiction
                kb.tell(logic.Expr("~", logic.PropSymbolExpr("P", w, h, 0)))

    # calculate successor states
    for t in range(t_max + 1):
        # action exclusion
        vacts = []
        for a in ["North", "South", "East", "West"]:
            ret = logic.PropSymbolExpr(a, t)
            ##print(ret)
            vacts.append(ret)
            ##print("action ", ret)
        # only one action from valid actions can be true
        kb.tell(exactlyOne(vacts)) # <<<< WRONG????
        ##print("Exactly ",exactlyOne(vacts))
        ##print(kb.clauses)
        if t > 0:
            for h in range(1, problem.getHeight() + 1):
                for w in range(1, problem.getWidth() + 1):
                    pac_pos = logic.PropSymbolExpr("P", w, h, t)
                    #kb.tell(pac_pos)
                    actions = problem.actions((w,h))
                    adjacent = []
                    for a in actions:
                        ##print("curr is: " + str((w,h)))
                        ##print("action is: " + str(a))
                        
                        up = (w, h + 1)
                        down = (w, h - 1)
                        right = (w + 1, h)
                        left = (w - 1, h)
                        north = logic.PropSymbolExpr("P", up[0], up[1], t - 1)
                        south = logic.PropSymbolExpr("P", down[0], down[1], t - 1)
                        east = logic.PropSymbolExpr("P", right[0], right[1], t - 1)
                        west = logic.PropSymbolExpr("P", left[0], left[1], t - 1)

                        # logic for successor state adjacent (action, state) pairs
                        act = None
                        direction = None
                        if a == "North":
                            direction = north
                            act = logic.PropSymbolExpr("South", t - 1)
                        elif a == "South":
                            direction = south
                            act = logic.PropSymbolExpr("North", t - 1)
                        elif a == "East":
                            direction = east
                            act = logic.PropSymbolExpr("West", t - 1)
                        elif a == "West":
                            direction = west
                            act = logic.PropSymbolExpr("East", t - 1)

                        result = logic.Expr("&", act, direction)
                        adjacent.append(result)
                    if adjacent:
                        options = logic.associate("|", adjacent)
                        result = logic.PropSymbolExpr("P", w, h, t)

                        implication = logic.Expr("<=>", result, options)
                        ##print(result, " <=> ", options)
                        kb.tell(implication)

        ##print("timestep ",t)
        #temp = kb.clauses
        temp = list(kb.clauses)
        # goal state cnf
        goal = problem.getGoalState()
        goal = logic.PropSymbolExpr("P", goal[0], goal[1], t)
        ##print("goal ",goal)
        temp.append(goal)
        ##print(temp)
        model = logic.pycoSAT(temp)
        ##print("model: " + str(model))
        if model:
            return extractActionSequence(model, ["North", "South", "East", "West"])
    return False


def foodLogicPlan(problem):
    """
    Given an instance of a FoodSearchProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    kb = logic.PropKB() # add axioms here
    t_max = 50
    start = problem.getStartState()[0]
    foodGrid = problem.getStartState()[1]
    flist = foodGrid.asList()
    ##print("food list: ", flist)

    # intial state; only 1 Pacman at intial time
    intial = logic.PropSymbolExpr("P", start[0], start[1], 0)
    kb.tell(intial)
    for h in range(1, problem.getHeight() + 1):
        for w in range(1, problem.getWidth() + 1):
            if (w,h) != start: # no contradiction
                kb.tell(logic.Expr("~", logic.PropSymbolExpr("P", w, h, 0)))

    # calculate successor states
    for t in range(t_max + 1):
        # action exclusion
        vacts = []
        for a in ["North", "South", "East", "West"]:
            ret = logic.PropSymbolExpr(a, t)
            ##print(ret)
            vacts.append(ret)
            ##print("action ", ret)
        # only one action from valid actions can be true
        kb.tell(exactlyOne(vacts)) # <<<< WRONG????
        ##print("Exactly ",exactlyOne(vacts))
        ##print(kb.clauses)
        if t > 0:
            for h in range(1, problem.getHeight() + 1):
                for w in range(1, problem.getWidth() + 1):
                    pac_pos = logic.PropSymbolExpr("P", w, h, t)
                    #kb.tell(pac_pos)
                    actions = problem.actions(((w,h), foodGrid))
                    adjacent = []
                    for a in actions:
                        ##print("curr is: " + str((w,h)))
                        ##print("action is: " + str(a))
                        
                        up = (w, h + 1)
                        down = (w, h - 1)
                        right = (w + 1, h)
                        left = (w - 1, h)
                        north = logic.PropSymbolExpr("P", up[0], up[1], t - 1)
                        south = logic.PropSymbolExpr("P", down[0], down[1], t - 1)
                        east = logic.PropSymbolExpr("P", right[0], right[1], t - 1)
                        west = logic.PropSymbolExpr("P", left[0], left[1], t - 1)

                        # logic for successor state adjacent (action, state) pairs
                        act = None
                        direction = None
                        if a == "North":
                            direction = north
                            act = logic.PropSymbolExpr("South", t - 1)
                        elif a == "South":
                            direction = south
                            act = logic.PropSymbolExpr("North", t - 1)
                        elif a == "East":
                            direction = east
                            act = logic.PropSymbolExpr("West", t - 1)
                        elif a == "West":
                            direction = west
                            act = logic.PropSymbolExpr("East", t - 1)

                        result = logic.Expr("&", act, direction)
                        adjacent.append(result)
                    if adjacent:
                        options = logic.associate("|", adjacent)
                        result = logic.PropSymbolExpr("P", w, h, t)

                        implication = logic.Expr("<=>", result, options)
                        ##print(result, " <=> ", options)
                        kb.tell(implication)


        temp = list(kb.clauses)
        # food search axioms
        for food in flist:
            foodLogic = []
            for tstep in range(0, t + 1):
                item = logic.PropSymbolExpr("P", food[0], food[1], tstep)
                ##print(item)
                foodLogic.append(item)

            eo = atLeastOne(foodLogic)
            temp.append(eo)
        
        model = logic.pycoSAT(temp)
        ##print("model: " + str(model))
        if model:
            return extractActionSequence(model, ["North", "South", "East", "West"])
    return False


def foodGhostLogicPlan(problem):
    """
    Given an instance of a FoodGhostSearchProblem, return a list of actions that help Pacman
    eat all of the food and avoid patrolling ghosts.
    Ghosts only move east and west. They always start by moving East, unless they start next to
    and eastern wall. 
    Available actions are game.Directions.{NORTH,SOUTH,EAST,WEST}
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"
    kb = logic.PropKB() # add axioms here
    t_max = 50
    start = problem.getStartState()[0]

    # fetch food grid
    foodGrid = problem.getStartState()[1]
    flist = foodGrid.asList()

    # fetch ghost states
    ghost_state = problem.getGhostStartStates()
    ghost_states = []

    # timestep 0 case
    for i in ghost_state:
        g = i.getPosition()
        ghost_states.append(g)
        initial = logic.Expr("~", logic.PropSymbolExpr("P", g[0], g[1], 1))
        kb.tell(initial)
    #print("ghosts positions: ", ghost_states)


    # intial state; only 1 Pacman at intial time
    intial = logic.PropSymbolExpr("P", start[0], start[1], 0)
    kb.tell(intial)
    for h in range(1, problem.getHeight() + 1):
        for w in range(1, problem.getWidth() + 1):
            if (w,h) != start: # no contradiction
                kb.tell(logic.Expr("~", logic.PropSymbolExpr("P", w, h, 0)))
    #print("expr list after intialization w/ ghosts: ", kb.clauses)

    # calculate successor states
    for t in range(t_max + 1):
        ##print("timestep: ", t)
        # action exclusion
        vacts = []
        for a in ["North", "South", "East", "West"]:
            ret = logic.PropSymbolExpr(a, t)
            ##print(ret)
            vacts.append(ret)
            ##print("action ", ret)
        # only one action from valid actions can be true
        kb.tell(exactlyOne(vacts)) # <<<< WRONG????
        ##print("Exactly ",exactlyOne(vacts))
        ##print(kb.clauses)
        if t > 0:
            for h in range(1, problem.getHeight() + 1):
                for w in range(1, problem.getWidth() + 1):
                    pac_pos = logic.PropSymbolExpr("P", w, h, t)
                    #kb.tell(pac_pos)
                    actions = problem.actions(((w,h), foodGrid))
                    adjacent = []
                    for a in actions:
                        ##print("curr is: " + str((w,h)))
                        ##print("action is: " + str(a))
                        
                        up = (w, h + 1)
                        down = (w, h - 1)
                        right = (w + 1, h)
                        left = (w - 1, h)
                        north = logic.PropSymbolExpr("P", up[0], up[1], t - 1)
                        south = logic.PropSymbolExpr("P", down[0], down[1], t - 1)
                        east = logic.PropSymbolExpr("P", right[0], right[1], t - 1)
                        west = logic.PropSymbolExpr("P", left[0], left[1], t - 1)

                        # logic for successor state adjacent (action, state) pairs
                        act = None
                        direction = None
                        if a == "North":
                            direction = north
                            act = logic.PropSymbolExpr("South", t - 1)
                        elif a == "South":
                            direction = south
                            act = logic.PropSymbolExpr("North", t - 1)
                        elif a == "East":
                            direction = east
                            act = logic.PropSymbolExpr("West", t - 1)
                        elif a == "West":
                            direction = west
                            act = logic.PropSymbolExpr("East", t - 1)

                        result = logic.Expr("&", act, direction)
                        adjacent.append(result)
                    if adjacent:
                        options = logic.associate("|", adjacent)
                        result = logic.PropSymbolExpr("P", w, h, t)
                        implication = logic.Expr("<=>", result, options)
                        ##print(result, " <=> ", options)
                        kb.tell(implication)

        temp = list(kb.clauses)
        # food search axioms
        for food in flist:
            foodLogic = []
            for tstep in range(0, t + 1):
                item = logic.PropSymbolExpr("P", food[0], food[1], tstep)
                ##print(item)
                foodLogic.append(item)

            eo = atLeastOne(foodLogic)
            temp.append(eo)

        if t > 0:
            # track ghost actions
            for g in ghost_states:
                gpos1 = ghostPostion(problem, g, t - 1)
                gpos2 = ghostPostion(problem, g, t)
                ##print("gpos1: ", gpos1)
                ##print("gpos2: ", gpos2)
                ghost1 = logic.Expr("~", logic.PropSymbolExpr("P", gpos1[0], gpos1[1], t))
                ghost2 = logic.Expr("~", logic.PropSymbolExpr("P", gpos2[0], gpos2[1], t))
                ##print("pac can't be at " + str(ghost1) + " at time ", t)
                ##print("pac can't be at " + str(ghost2) + " at time ", t)
                kb.tell(ghost1)
                kb.tell(ghost2)
                #temp.append(ghost1)
                #temp.append(ghost2)
        
        model = logic.pycoSAT(temp)
        ##print("model: " + str(model))
        if model:
            return extractActionSequence(model, ["North", "South", "East", "West"])
    return False
    
# return the location (x,y) of a ghost at a specific time step
# then add this position to where Pacman CANT be at to the kb
# ginit: initial position of the ghost
# time: the time step we want to know the location of the ghost at
def ghostPostion(problem, ginit, time):
    #ghosts can only move east or west; starts moving east
    direction = "east"
    currGhostPos = ginit
    for t in range(0, time):
        east = (currGhostPos[0] + 1, currGhostPos[1])
        west = (currGhostPos[0] - 1, currGhostPos[1])
        if problem.isWall(east):
            direction = "west"
        elif problem.isWall(west):
            direction = "east"
        if direction == "east":
            currGhostPos = east
        else:
            currGhostPos = west
    return currGhostPos

# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan
fglp = foodGhostLogicPlan

# Some for the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)