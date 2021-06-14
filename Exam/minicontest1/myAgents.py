# myAgents.py
# ---------------
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

from game import Agent
from searchProblems import PositionSearchProblem

import util
import time
import search
import math

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

path_travers = [] 
destination = []

class MyAgent(Agent):
    """
    Implementation of your agent.
    """

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        "*** YOUR CODE HERE ***"
        global path_travers
        global destination
        start_node = state.getPacmanPosition(self.index)
        food = state.getFood()
        q = AnyFoodSearchProblem(state, self.index)
        destination_x = destination[self.index][0]
        destination_y = destination[self.index][1]
        end_flag = False
        width = food.width
        height = food.height
        if (destination_x, destination_y) == (-1,-1) or food[destination_x][destination_y] is False or len(path_travers[self.index]) < 2:
            for counter_1 in range(width):
                for counter_2 in range(height):
                    if food[counter_1][counter_2]:
                        c = 0
                        path = search.bfs(q)
                        path_travers[self.index] = path
                        destination[self.index] = (counter_1, counter_2)
                        for counter_1 in range(len(destination)):
                            if destination[counter_1] == (counter_1, counter_2):
                                c = c+1
                        if c < 2:
                            end_flag = True
                        else:
                            end_flag = False
                    if end_flag:
                        break
        else:
            del(path_travers[self.index][0])
        return path_travers[self.index][0]

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        global path_travers
        global destination
        path_travers = [[1,] for i in range(10)]
        destination = [[-1,-1] for i in range(10)]

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""

class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"
        return search.aStarSearch(problem)

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

def manhattanDistance(state_A, state_B):
    if state_A[0] > state_B[0]:
        first_temp = state_A[0] - state_B[0]
    else:
        first_temp = state_B[0] - state_A[0]
    if state_A[1] > state_B[1]:
        second_temp = state_A[1] - state_B[1]
    else:
        second_temp = state_B[1] - state_A[1]
    temp1 = first_temp*first_temp
    temp2 = second_temp*second_temp
    sum_dist = (first_temp) ** 2 + (second_temp) ** 2
    final_result = math.sqrt(sum_dist)
    return final_result

def closestNode(fromNode, nodeList):
    if len(nodeList) == 0:
        return None
    minimum_Cost = manhattanDistance(fromNode, nodeList[0])
    closest_distance = nodeList[0]
    for state in nodeList[1:]:
        temp_cost = manhattanDistance(fromNode, state)
        if minimum_Cost > temp_cost:
            minimum_Cost = temp_cost
            closest_distance = state
    return closest_distance

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        if self.food[x][y] == True:
            return True
        else:
            return False

