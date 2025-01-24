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
import random, util, sys
import numpy as np

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
        newPos = successorGameState.getPacmanPosition()      # Pacman position after moving
        newFood = successorGameState.getFood()               # Remaining food
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        listFood = newFood.asList()                        # All remaining food as list
        ghostPos = successorGameState.getGhostPositions()  # Get the ghost position
        # Initialize with list 
        mFoodDist = []
        mGhostDist = []

        # Find the distance of all the foods to the pacman 
        for food in listFood:
          mFoodDist.append(manhattanDistance(food, newPos))

        # Find the distance of all the ghost to the pacman
        for ghost in ghostPos:
          mGhostDist.append(manhattanDistance(ghost, newPos))

        if currentGameState.getPacmanPosition() == newPos:
          return (-(float("inf")))

        for ghostDistance in mGhostDist:
          if ghostDistance < 2:
            return (-(float("inf")))

        if len(mFoodDist) == 0:
          return float("inf")
        else:
          minFoodDist = min(mFoodDist)
          maxFoodDist = max(mFoodDist)

        return 1000/sum(mFoodDist) + 10000/len(mFoodDist)


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

    """
        Your improved evaluation function here
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

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Here is the place to define your MiniMax Algorithm
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
        
     
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action
                            in possibleActions]
        max_action = max(action_scores)
        # if multiple moves give same score, pick a random one
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]
        
    def minimax(self, agent, depth, gameState):
      if gameState.isLose() or gameState.isWin() or depth == self.depth:
          return self.evaluationFunction(gameState)
      if agent == 0:  # maximize for pacman
          return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
                    getLegalActionsNoStop(0, gameState))
      else:  # minimize for ghosts
          nextAgent = agent + 1  # get the next agent
          if gameState.getNumAgents() == nextAgent:
              nextAgent = 0
          if nextAgent == 0:  # increase depth every time all agents have moved
              depth += 1
          return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                    getLegalActionsNoStop(agent, gameState))


class MinimaxAgent2(MultiAgentSearchAgent):
    """
      Here is the place to define your MiniMax Algorithm
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.transposition_table = {}

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]

    def minimax(self, agent, depth, gameState):
        # Check if the current state is already evaluated
        state_key = gameState.getHash()  # Generate a hash key for the state
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        # Terminal conditions
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pacman's turn (maximize)
            value = float('-inf')
            for action in getLegalActionsNoStop(agent, gameState):
                successor = gameState.generateSuccessor(agent, action)
                value = max(value, self.minimax(1, depth, successor))
            self.transposition_table[state_key] = value  # Store the value in the transposition table
            return value
        else:  # Ghost's turn (minimize)
            value = float('inf')
            nextAgent = agent + 1  # get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # increase depth every time all agents have moved
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                successor = gameState.generateSuccessor(agent, action)
                value = min(value, self.minimax(nextAgent, depth, successor))
            self.transposition_table[state_key] = value  # Store the value in the transposition table
            return value






class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Here is the place to define your Alpha-Beta Pruning Algorithm
    """

    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:  # maximize for pacman
            value = -999999
            for action in getLegalActionsNoStop(agent, gameState):
                value = max(value, self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            return value
        else:  # minimize for ghosts
            nextAgent = agent + 1  # get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # increase depth every time all agents have moved
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                value = 999999
                value = min(value, self.alphabeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            return value

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction using alpha-beta pruning.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        alpha = -999999
        beta = 999999
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]


import heapq

class AStarMinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax Agent with A* pathfinding for Pacman.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.transposition_table = {}
        self.last_positions = []

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]

    def minimax(self, agent, depth, gameState):
        # Check if the current state is already evaluated
        state_key = gameState.getHash()  # Generate a hash key for the state
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        # Terminal conditions
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pacman's turn (maximize)
            value = float('-inf')
            pacman_position = gameState.getPacmanPosition()
            food_positions = gameState.getFood().asList()

            # Use A* to guide Pacman toward the nearest food
            nearest_food = min(food_positions, key=lambda food: manhattanDistance(pacman_position, food))
            path_to_food = self.a_star(gameState, pacman_position, nearest_food)

            if path_to_food:  # If there's a valid path to the nearest food
                best_action = None
                if not self.avoid_repeated_positions(path_to_food[1]):
                    for action in getLegalActionsNoStop(agent, gameState):
                        successor = gameState.generateSuccessor(agent, action)
                        if successor.getPacmanPosition() == path_to_food[1]:
                            best_action = action
                            break
                
                if best_action:  # Use A* path as a heuristic guide
                    return self.evaluationFunction(gameState.generateSuccessor(agent, best_action))

            # If A* doesn't find a path, fallback to Minimax
            for action in getLegalActionsNoStop(agent, gameState):
                successor = gameState.generateSuccessor(agent, action)
                value = max(value, self.minimax(1, depth, successor))

            self.transposition_table[state_key] = value  # Store the value in the transposition table
            return value

        else:  # Ghost's turn (minimize)
            value = float('inf')
            nextAgent = agent + 1  # Get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # Increase depth every time all agents have moved
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                successor = gameState.generateSuccessor(agent, action)
                value = min(value, self.minimax(nextAgent, depth, successor))

            self.transposition_table[state_key] = value  # Store the value in the transposition table
            return value

    def a_star(self, gameState, start, goal):
        """
        A* search algorithm to find the shortest path from Pacman's current position to the goal (e.g., food).
        """
        # treated as a min heap
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        # start node to current node
        g_score = {start: 0}
        # estimated cost from start node to goal node
        f_score = {start: manhattanDistance(start, goal)}

        while open_list:
            current = heapq.heappop(open_list)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.getNeighbors(current, gameState):
                tentative_g_score = g_score[current] + 1  # Distance between neighbors is 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + manhattanDistance(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []
    
    # return the optimal path
    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def getNeighbors(self, position, gameState):
        """
        Returns the neighboring positions of the current position, considering walls and ghosts as obstacles.
        """
        x, y = position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        # Get ghost positions
        ghost_positions = [gameState.getGhostPosition(i) for i in range(1, gameState.getNumAgents())]

        # Filter out positions that are either walls or occupied by ghosts
        valid_neighbors = [neighbor for neighbor in neighbors 
                        if not gameState.hasWall(neighbor[0], neighbor[1]) and neighbor not in ghost_positions]
        
        return valid_neighbors
    
    def avoid_repeated_positions(self, current_position):
        if current_position in self.last_positions:
            return True  # Detected a loop
        self.last_positions.append(current_position)
        if len(self.last_positions) > 4:  # Keep track of the last 4 positions
            self.last_positions.pop(0)
        return False
    




def manhattanDistance(pos1, pos2):
    """
    Calculate the Manhattan distance between two positions.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])



class AStarAlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta Pruning Agent with A* pathfinding for Pacman.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.transposition_table = {}
        self.last_positions = []

    def getAction(self, gameState):
        """
        Returns the best action from the current gameState using Alpha-Beta pruning.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        alpha = float('-inf')
        beta = float('inf')
        action_scores = []

        for action in possibleActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.alphabeta(1, 0, successor, alpha, beta)  # Ghost starts at agent index 1
            action_scores.append(score)
            alpha = max(alpha, score)

        # Choose the action with the highest score
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]

    def alphabeta(self, agent, depth, gameState, alpha, beta):
        """
        Alpha-Beta pruning implementation.
        """
        # Check if the state is cached in the transposition table
        state_key = gameState.getHash()  # Generate a hash key for the state
        if state_key in self.transposition_table:
            return self.transposition_table[state_key]

        # Terminal conditions
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)

        if agent == 0:  # Pacman's turn (maximize)
            value = float('-inf')
            pacman_position = gameState.getPacmanPosition()
            food_positions = gameState.getFood().asList()

            # Use A* to guide Pacman toward the nearest food
            nearest_food = min(food_positions, key=lambda food: manhattanDistance(pacman_position, food))
            path_to_food = self.a_star(gameState, pacman_position, nearest_food)

            if path_to_food:  # If there's a valid path to the nearest food
                best_action = None
                if not self.avoid_repeated_positions(path_to_food[1]):
                    for action in getLegalActionsNoStop(agent, gameState):
                        successor = gameState.generateSuccessor(agent, action)
                        if successor.getPacmanPosition() == path_to_food[1]:
                            best_action = action
                            break

                if best_action:  # Use A* path as a heuristic guide
                    return self.evaluationFunction(gameState.generateSuccessor(agent, best_action))

            for action in getLegalActionsNoStop(agent, gameState):
                successor = gameState.generateSuccessor(agent, action)
                value = max(value, self.alphabeta(1, depth, successor, alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            self.transposition_table[state_key] = value
            return value

        else:  # Ghosts' turn (minimize)
            value = float('inf')
            nextAgent = agent + 1
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # Pacman will move after all ghosts, so increase depth
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                successor = gameState.generateSuccessor(agent, action)
                value = min(value, self.alphabeta(nextAgent, depth, successor, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            self.transposition_table[state_key] = value
            return value

    def a_star(self, gameState, start, goal):
        """
        A* search algorithm to find the shortest path from Pacman's current position to the goal (e.g., food).
        """
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: manhattanDistance(start, goal)}

        while open_list:
            current = heapq.heappop(open_list)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.getNeighbors(current, gameState):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + manhattanDistance(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return []

    def reconstruct_path(self, came_from, current):
        """
        Reconstructs the path found by A* from the goal to the start.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def getNeighbors(self, position, gameState):
        """
        Returns the neighboring positions of the current position, considering walls and ghosts as obstacles.
        """
        x, y = position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        ghost_positions = [gameState.getGhostPosition(i) for i in range(1, gameState.getNumAgents())]

        return [neighbor for neighbor in neighbors if not gameState.hasWall(neighbor[0], neighbor[1]) and neighbor not in ghost_positions]

    def avoid_repeated_positions(self, current_position):
        """
        Avoid positions that have been visited recently to prevent loops.
        """
        if current_position in self.last_positions:
            return True  # Detected a loop
        self.last_positions.append(current_position)
        if len(self.last_positions) > 4:  # Keep track of the last 4 positions
            self.last_positions.pop(0)
        return False

def manhattanDistance(pos1, pos2):
    """
    Calculate the Manhattan distance between two positions.
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])




def betterEvaluationFunction(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    """Calculate distance to the nearest food"""
    newFoodList = np.array(newFood.asList())
    distanceToFood = [util.euclideanDistance(newPos, food) for food in newFoodList]
    min_food_distance = float('inf')
    if len(newFoodList) > 0:
        min_food_distance = distanceToFood[np.argmin(distanceToFood)]
    
    """Calculate distance to nearest power pellet"""
    powerPellets = currentGameState.getCapsules()
    min_power_pellet_distance = float('inf')
    if len(powerPellets) > 0:
        distanceToPowerPellet = [util.euclideanDistance(newPos, pellet) for pellet in powerPellets]
        min_power_pellet_distance = distanceToPowerPellet[np.argmin(distanceToPowerPellet)]

    """Calculate distances to nearest ghosts"""
    ghostPositions = np.array(currentGameState.getGhostPositions())
    min_ghost_distance = float('inf')
    closest_scared_ghost_distance = float('inf')  # Distance to the closest scared ghost
    if len(ghostPositions) > 0:
        distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
        min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
        
        # Find the nearest scared ghost distance
        scared_ghosts_distances = [distance for i, distance in enumerate(distanceToGhost) if newScaredTimes[i] > 0]
        if scared_ghosts_distances:
            closest_scared_ghost_distance = min(scared_ghosts_distances)

        # Avoid certain death
        if min_ghost_distance <= 1 and newScaredTimes[np.argmin(distanceToGhost)] == 0:
            return -999999

        # Eat a scared ghost
        if min_ghost_distance <= 1 and newScaredTimes[np.argmin(distanceToGhost)] > 0:
            return 999999

    # Calculate the score
    score = currentGameState.getScore()
    if min_food_distance > 0:
        score += 10* 1 / min_food_distance  # Closer to food is better
    if min_ghost_distance > 0:
        score += 0.02* min_ghost_distance  # Further from non-scared ghosts is better
    if min_power_pellet_distance < float('inf'):
        score += 1 / min_power_pellet_distance  # Closer to power pellets is better

    # After eating a power pellet, prioritize the closest scared ghost
    if min_power_pellet_distance == 0:  # Indicates Pacman has just eaten a power pellet
        score += 1 / closest_scared_ghost_distance  # Go after the closest scared ghost
    
    # Consider remaining food (pallets)
    remaining_pallets = len(newFood.asList())
    if remaining_pallets > 0:
        score += 100* 1 / remaining_pallets  # Closer to collecting remaining pallets is better

    return score



def evaluationFunction(currentGameState):

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()

    """Calculate distance to the nearest food"""
    newFoodList = np.array(newFood.asList())
    distanceToFood = [util.euclideanDistance(newPos, food) for food in newFoodList]
    min_food_distance = 0
    if len(newFoodList) > 0:
        min_food_distance = distanceToFood[np.argmin(distanceToFood)]

    """Calculate the distance to nearest ghost"""
    ghostPositions = np.array(currentGameState.getGhostPositions())
    if len(ghostPositions) > 0:
        distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
        min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
        # avoid certain death
        if min_ghost_distance <= 1:
            return -999999
    

    return min_food_distance-min_ghost_distance


def getLegalActionsNoStop(index, gameState):
    possibleActions = gameState.getLegalActions(index)
    if Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)
    return possibleActions


# Abbreviation
better = betterEvaluationFunction

