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
    
        states = self.mdp.getStates()
        for i in range(self.iterations):
          valuesCopy = self.values.copy()
          for state in states:
            finalValue = -1
            for action in self.mdp.getPossibleActions(state):
              currentValue = self.computeQValueFromValues(state,action)
              if finalValue == -1 or finalValue < currentValue:
                finalValue = currentValue
            if finalValue == -1:
              finalValue = 0
            valuesCopy[state] = finalValue
          
          self.values = valuesCopy

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
        
        ret = 0
        transStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        
        for transition in transStatesAndProbs:
            T = transition[0]
            probability = transition[1]
            reward = self.mdp.getReward(state, action, T)
            value = self.getValue(T)
            # Q(s, a) = T(s, a, s')[R(s, a, s') + aVk(s)]
            ret += probability * (reward + self.discount * value)
            
        return ret

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        
        # No legal actions
        if self.mdp.isTerminal(state):
            return None
            
        actionList = self.mdp.getPossibleActions(state)
        maxValue = self.getQValue(state, actionList[0])
        maxAction = actionList[0]
        for action in actionList[1:]:
            value = self.getQValue(state, action)
            if maxValue < value:
                maxValue = value
                maxAction = action
                
        return maxAction

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
        states = self.mdp.getStates()
        for index in range(self.iterations):
            state = states[index % len(states)]
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                maxValue = self.computeQValueFromValues(state, actions[0])
                for action in actions[1:]:
                    qValue = self.computeQValueFromValues(state, action)
                    if qValue > maxValue:
                        maxValue = qValue
                self.values[state] = maxValue

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
        previous = {}
        states = self.mdp.getStates()
        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                        if nextState in previous:
                            previous[nextState].append(state)
                        else:
                            previous[nextState] = [state]
        
        priorityQueue = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                values = []
                for action in self.mdp.getPossibleActions(state):
                    values.append(self.computeQValueFromValues(state, action))
                priorityQueue.update(state, -abs(max(values) - self.values[state]))

        i = 0
        while i < self.iterations and not priorityQueue.isEmpty():
            state = priorityQueue.pop()
            if not self.mdp.isTerminal(state):
                values = []
                for action in self.mdp.getPossibleActions(state):
                    values.append(self.computeQValueFromValues(state, action))
                self.values[state] = max(values)
                
            for prev in previous[state]:
                if not self.mdp.isTerminal(prev):
                    values = []
                    for action in self.mdp.getPossibleActions(prev):
                        values.append(self.computeQValueFromValues(prev, action))
                    difference = abs(max(values) - self.values[prev])
                    if difference > self.theta:
                        priorityQueue.update(prev, -difference)
            i += 1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

