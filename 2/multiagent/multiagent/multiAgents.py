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
import math

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        food_list = currentGameState.getFood().asList()
        final_point = 0
        counter = 0
        for state in newGhostStates:
            directions = state.getPosition()
            distance = abs(newPos[0] - directions[0]) + abs(newPos[1] - directions[1])
            if newPos in food_list:
                final_point += 1
            if currentGameState.hasWall(newPos[0], newPos[1]):
                final_point -= 2
            if distance <= newScaredTimes[counter]:
                final_point += distance
            if distance < 2:
                final_point -= 2
            food_distance = []
            for c, d in food_list:
                howFar = abs(newPos[0] - c)
                food_distance.append(howFar)
            final_point -= 0.1 * min(food_distance)
            counter += 1
        return final_point


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        def value(state, agent, dp):
            if state.isWin() or state.isLose() or dp == 0:
                return self.evaluationFunction(state)
            if agent == 0:
                return maxvalue(state, agent, dp)
            else:
                return minvalue(state, agent, dp)

        def minvalue(state, agent_num, dp):
            action_list = state.getLegalActions(agent_num)
            successor_list = [state.generateSuccessor(agent_num, action) for action in action_list]
            minimum_value = float('inf')
            for successor in successor_list:
                minimum_value = min(minimum_value, value(successor, (agent_num + 1) % state.getNumAgents(), dp - 1))
            return minimum_value

        def maxvalue(state, agent_num, dp):
            action_list = state.getLegalActions(agent_num)
            successor_list = [state.generateSuccessor(agent_num, action) for action in action_list]
            maximum_value = -float('inf')
            for successor in successor_list:
                maximum_value = max(maximum_value, value(successor, (agent_num + 1) % state.getNumAgents(), dp - 1))
            return maximum_value

        dp = self.depth * gameState.getNumAgents()
        action_list = gameState.getLegalActions(0)
        scores = [value(gameState.generateSuccessor(0, action), 1, dp - 1) for action in action_list]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)
        return action_list[chosen_index]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def value(self, state, agent_num, dp, alpha, beta):
            if state.isWin() or state.isLose() or dp == 0:
                return self.evaluationFunction(state)
            if state.getNumAgents() == agent_num:
                return value(state, agent_num, dp - 1, alpha, beta)
            action_list = state.getLegalActions()
            num_of_elements = len(action_list)
            if num_of_elements > 0:
                if agent_num == 0:
                    best_choice = None
                    maximum_value = -float('inf')
                    for action in action_list:
                        val = self.value(self, state.generateSuccessor(agent_num, action),
                                         (agent_num + 1) % state.getNumAgents(), dp - 1, alpha, beta)
                        if val > maximum_value:
                            best_choice = action
                            maximum_value = val
                        alpha = max(val, alpha)
                        if alpha >= beta:
                            break
                    if self.depth == dp:
                        return best_choice
                    else:
                        return maximum_value
                    return maximum_value
                else:
                    action_list = state.getLegalActions(agent_num)
                    minimum_value = float('inf')
                    for action in action_list:
                        val = self.value(self, state.generateSuccessor(agent_num, action),
                                         (agent_num + 1) % state.getNumAgents(), dp - 1, alpha, beta)
                        if val < minimum_value:
                            minimum_value = val
                        beta = min(beta, val)
                        if alpha >= beta:
                            break
                    return minimum_value
            else:
                return self.value(self, state, dp, -float('inf'), float('inf'))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBetaSearch(self, gameState, depth, alpha, beta, agentIndex):
            if 0 == depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            if agentIndex == gameState.getNumAgents():
                return alphaBetaSearch(self, gameState, depth - 1, alpha, beta, 0)

            actions = gameState.getLegalActions(agentIndex)
            if 0 < len(actions):
                if 0 == agentIndex:
                    maximum = -1000000
                    bestAction = None
                    for action in actions:
                        value = alphaBetaSearch(self, gameState.generateSuccessor(agentIndex, action), depth, alpha,
                                                beta, agentIndex + 1)
                        if maximum < value:
                            maximum = value
                            bestAction = action
                        alpha = max(value, alpha)
                        if beta < alpha:
                            break

                    if self.depth == depth:
                        return bestAction
                    else:
                        return maximum

                else:
                    minimum = 1000000
                    for action in actions:
                        value = alphaBetaSearch(self, gameState.generateSuccessor(agentIndex, action), depth, alpha,
                                                beta, agentIndex + 1)
                        minimum = min(value, minimum)
                        beta = min(value, beta)
                        if beta < alpha:
                            break
                    return minimum
            else:
                return self.evaluationFunction(gameState)

        return alphaBetaSearch(self, gameState, self.depth, -1000000, 1000000, 0)


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

        def value(state, agent, dp):
            if state.isWin() or state.isLose() or dp == 0:
                return self.evaluationFunction(state)
            if agent == 0:
                return maxvalue(state, agent, dp)
            else:
                return expvalue(state, agent, dp)

        def expvalue(state, agent_num, dp):
            action_list = state.getLegalActions(agent_num)
            successor_list = [state.generateSuccessor(agent_num, action) for action in action_list]
            val = float(0)
            for successor in successor_list:
                val += value(successor, (agent_num + 1) % state.getNumAgents(), dp - 1)
            return val/len(successor_list)

        def maxvalue(state, agent_num, dp):
            action_list = state.getLegalActions(agent_num)
            successor_list = [state.generateSuccessor(agent_num, action) for action in action_list]
            maximum_value = -float('inf')
            for successor in successor_list:
                maximum_value = max(maximum_value, value(successor, (agent_num + 1) % state.getNumAgents(), dp - 1))
            return maximum_value

        dp = self.depth * gameState.getNumAgents()
        action_list = gameState.getLegalActions(0)
        scores = [value(gameState.generateSuccessor(0, action), 1, dp - 1) for action in action_list]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)
        return action_list[chosen_index]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    foodList = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    food_distance = []
    ghost_distance = []
    Scared_ghosts = 0
    capsuleList = currentGameState.getCapsules()
    pacman_position = list(currentGameState.getPacmanPosition())
    for food in foodList:
        direction_x = abs(food[0] - pacman_position[0])
        direction_y = abs(food[1] - pacman_position[1])
        food_distance.append(-1 * (direction_x + direction_y))
    for state in ghost_states:
        if state.scaredTimer is 0:
            Scared_ghosts += 1
            ghost_distance.append(0)
            continue
        ghost_position = state.getPosition()
        direction_x = abs(ghost_position[0] - pacman_position[0])
        directions_y = abs(ghost_position[1] - pacman_position[1])
        if (direction_x + directions_y) is 0:
            ghost_distance.append(0)
        else:
            ghost_distance.append(-1.0 / (direction_x + directions_y))
    if not food_distance:
        food_distance.append(0)
    return max(food_distance) + min(ghost_distance) + currentGameState.getScore() - 100 * len(
        capsuleList) - 20 * (len(ghost_states) - Scared_ghosts)


# Abbreviation
better = betterEvaluationFunction
