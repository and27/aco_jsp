from jsp import ACOp, GraphJSP, JSPNode
import numpy as np

#Read standard instances JSP
raw_instance_data = []
job_op = []  #Operation Time
job_ma = []  #Machines

with open("instances/abz6") as f:
    lines = f.readlines()
    mtd = lines[4].strip().split(" ") # mt-metadata (number of jobs and machines)
    for i in range (5, len(lines)):
        raw_line = lines[i].strip().replace("  ", " ").split(" ")
        raw_instance_data.append(raw_line) 
    
jobs_n = int(mtd[0])
jobs_m = int(mtd[1])

#Matriz de adyacencia para colocar los nodos que puedan ser visitados(se actualizará dinámicamente)
adj = np.zeros((jobs_n, jobs_m))

for i in range(jobs_n):
    single_job_op = []
    single_job_ma = []

    for j in range(jobs_m*2):
        if (j%2)==1:
            single_job_op.append(int(raw_instance_data[i][j]))
        else:
            single_job_ma.append(int(raw_instance_data[i][j]))

    job_op.append(single_job_op)
    job_ma.append(single_job_ma)
    
#counter for assignid each node an id
c = 0
nodes = []

#Here we are creating instances of the class JSPNode for all the possible nodes, to assign them the id, op_time, etc.
for i in range (jobs_n ):
    for j in range(jobs_m): 
        single_node = JSPNode(c, job_op[i][j], job_ma[i][j], i, jobs_m, jobs_n)
        #Here we append each created node to the master list "nodes"
        nodes.append(single_node)
        c+=1
        
#Due to "nodes" is a list we are rearragig it in a form of a matrix, were jobs are rows and cols are machines
nodes_matrix = np.array(nodes).reshape((jobs_n,jobs_m))

#Now we also create the instance of the whole graph as:
graph = GraphJSP(jobs_n, jobs_m, nodes_matrix)

aco = ACOp(cont_ant=5, generations=10, alfa=1.0, beta=1.0, ro=0.5, Q=0.3)  

#Another alternative is:
aco.resolve(graph)
