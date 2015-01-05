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
        # print("self.mdp")
        # print(self.mdp)
        self.discount = discount
        # print("discount")
        # print(discount)
        self.iterations = iterations
        # print("self.iterations")
        # print(self.iterations)
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Part 1: Run value interations up to 100 and construct MDPs
        states =  self.mdp.getStates()
        for i in range(0,self.iterations):
          # store_values here then transfer to self.values afterwards
          store_values = util.Counter()
          # print("store_values")
          # print (store_values) 
          # max_value = -99999999 
          for state in states:
            # print "all the states!"
            # print states
            # ie: ['TERMINAL_STATE', (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), (2, 2), (2, 4), (3, 0), (3, 2), (3, 3), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]

            # Special case: Check for terminal state and update store_values accordingly 
            terminalState = mdp.isTerminal(state)
            # print("Am I a terminal state?")
            # print(terminalState)
            if(terminalState):
              store_values[state] = 0


            # Part 2: loop through all possible actions and each possible transition state for each action to determine value iteration value. 
            # Bellman formula: https://www.youtube.com/watch?v=1S-dw6Vt1l4
            # initialize best_value to a really high negative 
            # value and find the max 
            max_value = -99999999  
            possible_actions = self.mdp.getPossibleActions(state)
            listActions = list(possible_actions)
            i = 0
            while i < len(listActions):
              action = listActions[i]
              i= i + 1
              # max_value = -99999999  
              # print("possible actions")
              # print(possible_actions)
              # ie: ('north', 'west', 'south', 'east')
              transition_state = self.mdp.getTransitionStatesAndProbs(state,action)
              total_values = 0
              for next_pos, transition_model in transition_state:
                # print("transition_state")
                # print(transition_state)
                # transition_model = T(s,a,s')
                # ie: [((3, 0), 0.8), ((4, 1), 0.1), ((4, 0), 0.1)]
                reward = self.mdp.getReward(state,action,next_pos)
                # reward is always 0 when not a terminal state..
                # reward = R(s,a,s')
                # print "reward"
                # print reward

                future_rewards = self.discount*self.values[next_pos]
                # T(s,a,s') 
                total_values += transition_model*(reward+future_rewards)

              # find the max sum of values
              if max_value < total_values:
                max_value = total_values
                # update stored values
                store_values[state] = max_value
          self.values = store_values

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
        "*** YOUR CODE HERE ***"        
        # return synthesized policy pi_{k+1}
        q_iteration = 0
        transition_state = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_pos, transition_model in transition_state:
            reward = self.mdp.getReward(state, action, next_pos)
            state_reached = self.values[next_pos]
            discount_mul_state_reached = self.discount * state_reached
            # print("state_reached")
            # print(state_reached)
            q_iteration += transition_model * (reward + discount_mul_state_reached)
        # print("q_iteration")
        # print(q_iteration)
        return q_iteration

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # find the max value
        possibleActions = self.mdp.getPossibleActions(state)
        max_value = -9999999
        max_action = None
        terminalState = self.mdp.isTerminal(state)

        if(not terminalState):
          i = 0
          possible_actions = self.mdp.getPossibleActions(state)
          listActions = list(possible_actions)
          while i < len(listActions):
            action = listActions[i]
            i= i + 1
            q_value = self.computeQValueFromValues(state,action)
            if q_value > max_value:
              max_value = self.computeQValueFromValues(state,action)
              max_action = action
          return max_action
        # special case: when state is terminal 
        return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
