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
        "*** YOUR CODE HERE ***"
        lenght = self.iterations
        for iterate in range(lenght):
            values = util.Counter()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    action = self.getAction(state)
                    values[state] = self.computeQValueFromValues(state, action)
            self.values = values

    def getValue(self, state):
        """Return the value of the state (computed in __init__)."""
        return self.values[state]
        util.raiseNotDefined()


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        value = 0
        states = self.mdp.getTransitionStatesAndProbs(state, action)
        for (next_state, p) in states:
            gama = self.discount
            v = self.values[next_state]
            r = self.mdp.getReward(state, action, next_state)
            value = value + p*(r + v*gama)
        return value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        top_value = 0
        top_action = None
        for action in self.mdp.getPossibleActions(state):
            value = 0
            if top_value == 0 or top_value < self.computeQValueFromValues(state, action):
                top_value = self.computeQValueFromValues(state, action)
                top_action = action
        return top_action
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        lenght_of_states = len(states)
        lenght = self.iterations
        for iterate in range(lenght):
            index = iterate % lenght_of_states
            state = states[index]
            if not self.mdp.isTerminal(state):
                values = []
                allowed_actions = self.mdp.getPossibleActions(state)
                for action in allowed_actions:
                    QValue = self.computeQValueFromValues(state, action)
                    values.append(QValue)
                self.values[state] = max(values)


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
        "*** YOUR CODE HERE ***"
        p_queue = util.PriorityQueue()
        predecessors = {}
        states = self.mdp.getStates()
        for state in states:
            if not self.mdp.isTerminal(state):
                allowed_actions = self.mdp.getPossibleActions(state)
                for action in allowed_actions:
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                    for next_state, t in transitions:
                        if next_state in predecessors:
                            predecessors[next_state].add(state)
                        else:
                            predecessors[next_state] = {state}
        statesstar = self.mdp.getStates()                    
        for state in statesstar:
            if not self.mdp.isTerminal(state):
                values = []
                allowed_actions = self.mdp.getPossibleActions(state)
                for action in allowed_actions:
                    QValue = self.computeQValueFromValues(state, action)
                    values.append(QValue)
                top_value = max(values)
                if top_value > self.values[state]:
                    sub = top_value - self.values[state]
                else:
                    sub = self.values[state] - top_value
                p_queue.update(state, -sub)
        lenght = self.iterations
        for i in range(lenght):
            if p_queue.isEmpty():
                break
            pop_state = p_queue.pop()
            if not self.mdp.isTerminal(pop_state):
                values = []
                allowed_action = self.mdp.getPossibleActions(pop_state)
                for action in allowed_action:
                    QValue = self.computeQValueFromValues(pop_state, action)
                    values.append(QValue)
                self.values[pop_state] = max(values)

            for t in predecessors[pop_state]:
                if not self.mdp.isTerminal(t):
                    values = []
                    for action in self.mdp.getPossibleActions(t):
                        QValue = self.computeQValueFromValues(t, action)
                        values.append(QValue)
                    top_value = max(values)
                    if top_value > self.values[t]:
                        sub = top_value - self.values[t]
                    else:
                        sub = self.values[t] - top_value
                    if sub > self.theta:
                        p_queue.update(t, -sub)