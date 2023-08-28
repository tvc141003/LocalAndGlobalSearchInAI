import queue
import random
import sys
import time

def heuristic(state):
    queens = len(state)
    h = 0
    for queen in range(queens):
        index = 1
        while (queen - index >= 0):
            # Check row
            if (state[queen] == state[queen - index]): h += 1
            # Check diagonal
            if (state[queen] - index == state[queen - index] or state[queen] + index == state[queen - index]): h += 1
            index += 1
    
    return h

class Problem():
    def __init__(self, queens = 8, searching = 1):
        self._queens = queens
        self._search = searching
        self._initState = [0 for i in range(self._queens)]
        # self._initState = [random.randint(0, self._queens - 1) for i in range(self._queens)]

class Search():
    def __init__(self, numberOfQueen = 8, searching = 1):
        self._number = numberOfQueen
        self._search = searching

    def run(self, problem):
        pass

class UcsSearch(Search):
    def run(self, problem):
        frontier = queue.PriorityQueue()
        explored = []
        node = problem._initState
        pathCost = 0
        frontier.put((pathCost, node))
        while (not frontier.empty()):
            node = frontier.get()
            nodeCost = node[0]
            state = node[1]

            if (state in explored):
                continue

            # Goal state -> heuristic = 0
            if (heuristic(state) == 0): 
                # get data usage by byte
                dataUse = sys.getsizeof(explored) + sys.getsizeof(node)*frontier.qsize()
                # convert byte to Mb
                dataUse = dataUse / (1024**2)

                print(f"Data usage: {round(dataUse, 4)}Mb")
                return state
            # Add state to explored
            explored.append(state)
            
            # number = node.State
            number = len(state)
            for queen in range(number):
                for i in range(number):
                    childState = state[:]
                    if i != state[queen]:
                        childState[queen] = i
                    if (childState not in explored):
                        frontier.put((nodeCost+1, childState))


        if (frontier.empty()):
            raise "Failure"
        
class AStartSearch(Search):
    def run(self, problem):
        frontier = queue.PriorityQueue()
        explored = []
        node = problem._initState
        pathCost = 0
        frontier.put((pathCost + heuristic(node), node))
        while (not frontier.empty()):
            node = frontier.get()
            nodeCost = node[0]
            state = node[1]
            nodeCost -= heuristic(state)

            if (state in explored):
                continue

            # Goal state -> heuristic = 0
            if (heuristic(state) == 0): 
                # get data usage by byte
                dataUse = sys.getsizeof(explored) + sys.getsizeof(node)*frontier.qsize()
                # convert byte to Mb
                dataUse = dataUse / (1024**2)

                print(f"Data usage: {round(dataUse, 4)}Mb")
                return state
            # Add state to explored
            explored.append(state)
            
            # number = node.State
            number = len(state)
            for queen in range(number):
                for i in range(number):
                    childState = state[:]
                    if i != state[queen]:
                        childState[queen] = i
                    if (childState not in explored):
                        frontier.put((nodeCost + heuristic(childState) + 1, childState))


        if (frontier.empty()):
            raise "Failure"
        
class GeneticSearch(Search):
    def __init__(self):
        self._state = []
        self._queens = 0
        self._numberState = 100
    
    def initState(self, problem):
        self._queens = problem._queens
        for _ in range(self._numberState):
            state = [random.randint(0, problem._queens - 1) for _ in range(self._queens)]
            heu = heuristic(state)
            self._state.append((heu, state))

    # return a array with probability 
    def fitness(self, population):
        MAX_NON_ATTACKING = self._queens*(self._queens - 1) / 2

        sumNonAttack = 0
        for state in population:
            sumNonAttack += MAX_NON_ATTACKING - state[0]

        probability = [(MAX_NON_ATTACKING - state[0]) / sumNonAttack for state in population]
        return probability
    
    # return parent with high probability
    def selection(self, probability):
        parentsTuple = random.choices(self._state, weights = probability, k = 1)

        parent = parentsTuple[0][1]
        return parent
    
    # from parents return new Child
    def crossOver(self, parent1, parent2):
        position = random.randint(1, self._queens - 1)
        child = parent1[:position] + parent2[position:]

        return child
    
    def mutation(self, child):
        column = random.randint(0, self._queens - 1)
        row = random.randint(0, self._queens - 1)
        child[column] = row
        return child
    
    def reGeneral(self, newPopulation):
        childs = queue.PriorityQueue()
        parents = queue.PriorityQueue()
        for node in self._state:
            parents.put(node)
        for state in newPopulation:
            childs.put((heuristic(state), state))

        count = 0
        newGeneral = []
        while (count < self._numberState):
            if (random.random() < 0.3):
                newGeneral.append(parents.get())
            else:
                newGeneral.append(childs.get())
            count += 1 

        self._state = newGeneral
        
    def run(self, problem):
        self.initState(problem)
        while (True):
            for state in self._state:
                if (state[0] == 0):
                    # get data usage by byte
                    dataUse = 4*len(self._state)*(sys.getsizeof(self._state[0]) + sys.getsizeof(self._state[0][1]) + sys.getsizeof(self._state[0][0]))
                    # convert byte to Mb
                    dataUse = dataUse / (1024**2)

                    print(f"Data usage: {round(dataUse, 4)}Mb")
                    return state[1]
                
            newPopulation = []
            fitnessProbability = self.fitness(self._state)
            
            for _ in range(self._numberState):
                parent1 = self.selection(fitnessProbability)
                parent2 = self.selection(fitnessProbability)
                child = self.crossOver(parent1, parent2)
                
                if (random.random() < 0.8):
                    child = self.mutation(child)
                newPopulation.append(child)

            self.reGeneral(newPopulation)

def inputValue():
    queens = int(input('Input number of queens: '))
    search = int(input('Input type of Searching\n1: Uniform-cost search\n2: A*\n3: Genetic algorithm\n'))
    if (queens <= 0 or search < 0 or search > 4):
        raise "Invalid input"
    
    problem = Problem(queens = queens)
    typeSearch = Search()
    if search == 1:
        typeSearch = UcsSearch()
    elif search == 2:
        typeSearch = AStartSearch()
    else: typeSearch = GeneticSearch()

    return problem, typeSearch
        
if (__name__ == '__main__'):
    problem, search = inputValue()
    goal = []
    begin = time.time()
    goal = search.run(problem)
    end = time.time()
    print(f"Wall time: {round(end - begin, 7) * 100}ms")
    print(goal)
    for i in range(problem._queens):
        for j in range(problem._queens):
            if (goal[j] == i): print('Q', end = ' ')
            else: print('+', end = ' ')
        print("")