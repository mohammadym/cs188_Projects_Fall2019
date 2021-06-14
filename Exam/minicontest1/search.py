# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

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

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    visited_nodes = []
    path_of_nodes = []
    parent_path = {}
    stack = util.Stack()
    start_node = problem.getStartState()
    stack.push([start_node, "Stop", 0])
    while not stack.isEmpty():
        current_node = stack.pop()
        current_node_name = current_node[0]
        if problem.isGoalState(current_node_name):
            break
        else:
            if current_node_name not in visited_nodes:
                visited_nodes.append(current_node_name)
            else:
                continue
            actions = problem.getSuccessors(current_node_name)
            for action in actions:
                stack.push(action)
                parent_path[action] = current_node
    while current_node is not None:
        path_of_nodes.append(current_node[1])
        if current_node[0] != start_node:
            current_node = parent_path[current_node]
        else:
            current_node = None
    path_of_nodes.reverse()
    return path_of_nodes[1:]
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    visited_nodes = []
    path_of_nodes = []
    parent_path = {}
    start_node = problem.getStartState()
    queue = util.Queue()
    queue.push([start_node, "Stop", 0])
    while not queue.isEmpty():
        current_node = queue.pop()
        current_node_name = current_node[0]
        if problem.isGoalState(current_node_name):
            break
        else:
            if current_node_name not in visited_nodes:
                visited_nodes.append(current_node_name)
            else:
                continue
            actions = problem.getSuccessors(current_node_name)
            for action in actions:
                if action[0] not in visited_nodes:
                    queue.push(action)
                    parent_path[action] = current_node
    while current_node is not None:
        path_of_nodes.append(current_node[1])
        if current_node[0] != start_node:
            current_node = parent_path[current_node]
        else:
            current_node = None
    path_of_nodes.reverse()
    return path_of_nodes[1:]

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited_nodes = []
    path_of_nodes = []
    parent_path = {}
    p_queue = util.PriorityQueueWithFunction(lambda action: action[2])
    start_node = problem.getStartState()
    p_queue.push([start_node, "Stop", 0])
    while not p_queue.isEmpty():
        current_node = p_queue.pop()
        current_node_name = current_node[0]
        if problem.isGoalState(current_node_name):
            break
        else:
            if current_node_name not in visited_nodes:
                visited_nodes.append(current_node_name)
            else:
                continue
            actions = problem.getSuccessors(current_node_name)
            for action in actions:
                cost_of_travers = current_node[2] + action[2]
                if action[0] not in visited_nodes:
                    p_queue.push((action[0], action[1], cost_of_travers))
                    parent_path[(action[0], action[1])] = current_node
    while current_node is not None:
        path_of_nodes.append(current_node[1])
        if current_node[0] != start_node:
            current_node = parent_path[(current_node[0], current_node[1]
            )]
        else:
            current_node = None
    path_of_nodes.reverse()
    return path_of_nodes[1:]

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    visited_nodes = []
    path_of_nodes = []
    parent_path = {}
    start_node = problem.getStartState()
    p_queue = util.PriorityQueueWithFunction(lambda action: action[2] + heuristic(action[0], problem))
    p_queue.push((start_node, "Stop", 0))
    parent_path[(start_node, "Stop", 0)] = None
    while p_queue.isEmpty() is False:
        current_node = p_queue.pop()
        if problem.isGoalState(current_node[0]):
            break
        else:
            current_node_name = current_node[0]
            if current_node_name not in visited_nodes:
                visited_nodes.append(current_node_name)
            else:
                continue
            actions = problem.getSuccessors(current_node_name)
            for action in actions:
                cost_of_travers = current_node[2] + action[2]
                if action[0] not in visited_nodes:
                    p_queue.push((action[0], action[1], cost_of_travers))
                    parent_path[(action[0], action[1])] = current_node
    while current_node is not None:
        path_of_nodes.append(current_node[1])
        if current_node[0] != start_node:
            current_node = parent_path[(current_node[0], current_node[1])]
        else:
            current_node = None
    path_of_nodes.reverse()
    return path_of_nodes[1:]

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
