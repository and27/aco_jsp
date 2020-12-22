#import required libraries
import math
import sys
import argparse
from mpi4py import MPI
import random
import time
import concurrent.futures
import operator
import matplotlib.pyplot as plt
#from skimage import io
import numpy as np


# roulette_wheel and rank_selection are techniques for selecting the next node

def roulette_wheel(probabilities):
    seleccionado = -1
    rand = random.random()
    for i, probabilidade in enumerate(probabilities):
        rand -= probabilidade
        if rand <= 0:
            seleccionado = i
            break
    return seleccionado


def rank_selection(probabilities):
    tab = []
    prob = npy.array(probabilities)
    n = len(probabilities)
    for i in range(n):
        tab.append((i,probabilities[i]))
    n1 = len(prob.nonzero()[0])
    rank_sum = n1 * (n1 + 1) / 2
    sortedby2 = sorted(tab, key = lambda tup: tup[1])
    print(sortedby2)
    seleccionado = 0
    rand = random.random()
    print(rand)
    for i, probabilidade in enumerate(sortedby2):
        if probabilidade[1]!=0:
            rand-=((i+1)/rank_sum)
            print(rand)
            if rand <= 0:
                print("se ha seleccionado a: ")
                seleccionado = probabilidade[0]
                print(probabilidade)
                break
    return seleccionado


"""                
Lets create the ACOs class that will store the parameters of the algorithm 
and initialize the solution construction process and the global pheromone update
"""
class ACOp(object):
    def __init__(self, cont_ant, generations, alfa, beta, ro, Q=0.0):
        
        self.Q = Q                       #Pheromone intensity (default = 0.0)
        self.ro = ro                     #Evaporation ratio 
        self.beta = beta                 #Visibility influence (Nodos cercanos tienen mayor peso)
        self.alfa = alfa                 #Phromone influence
        self.cont_ant = cont_ant         #Ant counter 
        self.generations = generations   #Algorithm iterations
        
    def _update_pheromone(self, graph, ants):

        for i, row in enumerate(graph.pheromone):
            for j, column in enumerate(row):
                graph.pheromone[i][j] *= self.ro  # evaporation level
                for ant in ants:
                    
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]
                    #if(i==0 and j==1):
                        #print("when i == 0 and j ==1 ", ant.pheromone_delta[i][j])
                        #print("main pheromone in this position: ", graph.pheromone[i][j])
        
        #Pheromone debugging
        """
        print("\nDebugging phero main")
        for x in range(graph.total_nodes):
            for y in range(graph.total_nodes):
                print(round(graph.pheromone[x][y],3), end=' ')
            print("")
        """
            
    def resolve(self, graph):
        best_cost = float('inf')
        best_sol = []
        for g in range(self.generations):
            print("\nRunning generation: ", g)
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            _rank = comm.Get_rank()          
                
            ants = [_ant](self,graph) for i in range(int(self.cont_ant/size))
            remaining = self.cont_ant % size
            for r in range(1, remaining+1):
                if _rank == r:
                    ants.append(_ant(self,graph))

            for a in ants:           
                #Pheromone debugging
                """
                print("\nDebugging phero local")
                for x in range(graph.total_nodes):
                    for y in range(graph.total_nodes):
                        print(round(graph.pheromone[x][y],3), end=' ')
                    print("")
                """
                for i in range(graph.total_nodes-1):
                    a._next_node()
                print(" . ", end='')
                time.sleep(10/100)
                
                #for v in a.visited:
                    #print(v.id)
                if a.total_cost < best_cost:
                    best_cost = a.total_cost
                    print("\nNew best cost found: ", best_cost)   
                #print("makespan is:", max(a.machine_release_time))
                a._update_pheromone_delta()

            if _rank:
                comm.send(ants, dest=0, tag=1)
            else:
                for i in range(1,size):
                    for ant in comm.recv(source=i, tag=1)
                        ants.append(ant)
                self._update_pheromone(graph, ants)
       
        if not _rank:
            return best_cost, best_sol
                
"""                
Lets create the ant class that will store the 
information related to the visited path by each ant
"""
class _ant(object):
    #The constructor initializes the values corresponding to the matrices heu and pher. And algo obtain alpha, beta parameters
    #using the aco (colony instance).
    def __init__(self, aco, graph):
        #print("Initialize ant")
        
        self.colony = aco
        #graph now contain the nodes and no the adjacency
        self.graph = graph
        self.nodes = graph.nodes_matrix
        self.total_cost = 0.0  # Lk
        
        #Las operaciones de los jobs que ya estén consideras se añadirán a esta lista. 
        #Hay que tomar en cuenta que solo pueden ir añadiendose en orden (cada job tiene un orden en las ops)
        self.visited = []  
        
        #The folowing additional lists will hold the information related to the makespan of the jsp
        self.machine_state=[False]*graph.total_machines
        self.machine_release_time=[0]*graph.total_machines
        self.job_finishing_time=[0]*graph.total_jobs
                
        self.pheromone_delta = [] 
        
        #in the JSP case we need to modify the graph.adjacency dinamically to know the allowed nodes.
        #self.allow = graph.adj
        # en vez de guardar el valor de la operacion, deberíamos numerar los nodos y guardar su id en self.allow
        
        self.allow=[]
        for i in range(self.graph.total_jobs):
            #Here we append the first node of all jobs to the the only allowed for the first transition from node_00
            self.allow.append(self.nodes[i][0])
            #print("allowed node: ", self.allow[i].machine)
        
        #para jsp se pueden considerar varias metricas y variaciones para crear la matriz heurística
        self.eta = [
            [ self.nodes[i][j].op_time
                 for i in range (self.graph.total_machines) ] 
                 for j in range (self.graph.total_jobs)
            ]

        inicio = random.randint(0, len(self.allow)-1)  # random start
        start_node = self.allow[inicio]
        #print("starting node: ", inicio)
        
        #we first need to verify if it is not the las node (well in the future cases)
        #In this particular case we only select the next one, which correspods to column --> 1 ([1])
        #The job_id tell us what is the row of the current selected node (start_node)
        self.allow.append(self.nodes[start_node.job_id][1])
        
        #Now we pop out the node selected and append it to the visited ones.
        self.visited.append(self.allow.pop(inicio))
        #Where we use atual?
        self.atual = inicio
        
        #JSP part
        self.machine_release_time[start_node.machine]=start_node.op_time
        #print("machine release time", self.machine_release_time)
        self.machine_state[start_node.machine]=True
        self.job_finishing_time[start_node.job_id] = start_node.op_time
        #print("job finishing times: ", self.job_finishing_time)
        #print("machine releasing time: ", self.machine_release_time)
        #print("machine states: ", self.machine_state)
        
    #The next_node function selects the next node using probabilities (Roullete Wheel method used)
    def _next_node(self):
        #print ("node selection \n")
        denominador = 0
        
        #In this case, as the heuristic information (eta) is only the operation time, we can directly get it from the node.
        #We can consider another metrics as shown here: "Ant system for job-shop scheduling- paper"
        #However, the pheromones we can put them on a general (graph) object and also store there the total_jobs and total_m
        #we require the id of the actual node and the id of the allowed nodes (j.id)
        for j in self.allow:
            #print("node allowed: ", j.id)
            denominador += self.graph.pheromone[
                self.atual][j.id] **self.colony.alfa   *   j.op_time ** self.colony.beta
            
        #print("denominator", denominador)
        probabilidades = [
            0 for i in range(len(self.allow))
        ]  # initialize probability list according to number of nodes
        
        #Here we create the calc_prob function that will be executed concurrently below
        def calcula_probabilidades(i):
        #try:
            self.allow[i]
            probabilidades[i] = self.graph.pheromone[self.atual][self.allow[i].id] ** self.colony.alfa * \
                self.allow[i].op_time ** self.colony.beta / denominador
    
        #except ValueError:
         #   pass  # 

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            futures = [
                executor.submit(calcula_probabilidades, i)
                for i in range(len(self.allow))
            ]

        # select next node
        selecionado = 0
        rand = random.random()
        #print("Random value: ", rand)
        for i, probabilidade in enumerate(probabilidades):
            #print("Probabilidades: ", probabilidade)
            rand -= probabilidade
            if rand <= 0:
                selecionado = i
                break
                
        #print("\nindex seleccionado is: ", selecionado, end = ' ')
        #print("then it is: ", self.allow[selecionado].id)
        
        #When we select a node, it means we select an operation (we are interested on finishing all operations)
        node_selected = self.allow[selecionado]
        
        #now that we select the node we have to put the machine in a occupied state (true)
        node_ma = node_selected.machine
        node_job = node_selected.job_id
        
        #CASES from  A Modified Ant Colony Algorithm for the Job Shop Scheduling Problem to Minimize Makespan (Zhang)
        if (self.machine_state[node_ma] == True and self.job_finishing_time[node_job] != 0):
            self.job_finishing_time[node_job] = max(self.machine_release_time[node_ma] + node_selected.op_time, self.job_finishing_time[node_job]+node_selected.op_time)
            self.machine_release_time[node_ma] = self.job_finishing_time[node_job]    
          
        elif (self.machine_state[node_ma] == True and self.job_finishing_time[node_job] == 0):
            self.job_finishing_time[node_job] = self.machine_release_time[node_ma] + node_selected.op_time
            self.machine_release_time[node_ma] = self.job_finishing_time[node_job]    

        elif (self.machine_state[node_ma] == False and self.job_finishing_time[node_job] != 0):
            self.job_finishing_time[node_job] += node_selected.op_time
            self.machine_release_time[node_ma]=node_selected.op_time

        elif (self.machine_state[node_ma] == False and self.job_finishing_time[node_job] == 0):
            self.job_finishing_time[node_job] = node_selected.op_time
            self.machine_release_time[node_ma]=node_selected.op_time
        
        self.machine_state[node_selected.machine]=True

        #print("job finishing times: ", self.job_finishing_time)
        #print("machine releasing time: ", self.machine_release_time)
        #print("machine states: ", self.machine_state)
        
        #if is not the last operation in the job we need to allow the next one
        #In this case if the total machines is 20 and the actual node is 10 then 20%10 will be always 0
        #In this case 10 will be in the second line, so we do not have to add it but pass (ignore)
        if (node_selected.id+1) % node_selected.total_machines != 0:
            #print("nodeid: ", node_selected.id)
            #print("result", node_selected.id+1 % node_selected.total_machines)
            self.allow.append(self.nodes[node_selected.job_id][node_selected.id % self.graph.total_machines + 1])
        else:
            pass
        
        self.visited.append(self.allow[selecionado])
        
        #The total cost in jsp will sum the operation cost
        
        self.total_cost = max(self.job_finishing_time)
        self.atual = selecionado
        
        #In JSP we only need to provide the index of the selected between allowed nodes to delete it from this list
        del(self.allow[selecionado])
        
    
    #Now lets define a method that allow us to update the local pheromone matrix according to the path length
    def _update_pheromone_delta(self):
        
        self.pheromone_delta = [
            [0 for j in range(self.graph.total_nodes)]  
            for i in range(self.graph.total_nodes)
        ]
        for _ in range(1, len(self.visited)):
            i = self.visited[_ - 1].id
            #print("i is equal to: ", i)
            j = self.visited[_].id
            #print("j is equal to: ", j)
            self.pheromone_delta[i][j] = self.colony.Q / self.total_cost       
             
class GraphJSP(object):
    def __init__(self, total_machines, total_jobs, nodes_matrix):
        self.total_machines = total_machines
        self.total_jobs = total_jobs
        self.total_nodes = total_machines*total_jobs
        #self.pheromone = [[1 / (self.total_jobs * self.total_machines) for j in range(self.total_nodes)]
         #                 for i in range(self.total_nodes)]  # m x m
            
        self.pheromone = [[1 / (self.total_nodes * self.total_nodes) for j in range(self.total_nodes)]
                          for i in range(self.total_nodes)]  # m x m
        self.nodes_matrix = nodes_matrix
        
class JSPNode(object):
    def __init__(self, nid, op, mach, job, total_machines, total_jobs):
        self.id = nid
        self.op_time = op
        self.machine = mach
        self.job_id = job
        self.is_allowed = False
        self.total_machines = total_machines
        self.total_jobs = total_jobs

