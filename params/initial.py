# simulation lengths
T_horizon = None # time horizon (to be set later)
T_horizon_list = 2000*np.arange(1,11)
N_runs    = 50 # number of simulations

# noise and arrival rates
arrival_rate   = 5  # lambda
noise_variance = 5 # sigma^2

# algorithm parameters
beta  = 8*noise_variance # exploration tuner (beta > 4 sigma^2)
nu    = None # backlog-cost tradeoff tuner (T^{1/3}) (to be set later)
delta = None # exploration tuner (T^{(-2 sigma^2)/(beta - 2 sigma^2)}) (to be set later)

# topology
N_nodes          = 5
source_node      = 0
destination_node = 4
edges_list       = [[0,1], [0,2], [0,3], [1,4], [2,4], [3,4]]
node_edge_adjacency = mutil.prepare_adjacency(edges_list, N_nodes)

# edge properties
edge_capacities = np.array([1, 2, 3, 5, 5, 5])
true_edge_costs = np.array([1, 2, 3, 6, 4, 2])

# pack parameters
simulation_params = mutil.SimulationParameters(node_edge_adjacency, 
                 true_edge_costs, edge_capacities, 
                 source_node, destination_node, 
                 noise_variance, 
                 arrival_rate, 
                 N_runs, T_horizon, 
                 beta, delta, nu)