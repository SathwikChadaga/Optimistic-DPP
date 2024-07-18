import numpy as np
import utils.misc_utils as mutil
import utils.online_queueing_network as queuenetol
import utils.policies as polc

def run_horizon_experiment(simulation_params, policy = 'dpop', rates = 'planned'):
    # store algorithm parameters
    beta  = simulation_params.beta
    delta = simulation_params.delta
    nu    = simulation_params.nu

    # define network from the given parameters
    queueing_network = queuenetol.OnlineQueueNetwork(simulation_params)

    network_status = 0
    while(network_status == 0):
        # get queue state
        queue_state = queueing_network.queues[:, :, queueing_network.tt]

        # estimate edge costs from previous observation
        if(policy == 'dpop'): 
            estimated_edge_costs = queueing_network.edge_cost_means - np.sqrt(beta*np.log((queueing_network.tt+1)/delta)/queueing_network.edge_num_pulls)
        else: # policy == 'oracle' 
            estimated_edge_costs = queueing_network.true_edge_costs
        estimated_cost_matrix = mutil.prepare_cost_matrix(queueing_network.node_edge_adjacency, estimated_edge_costs)

        # get planned transmissions from the policy
        planned_edge_rates = polc.max_weight_policy(queue_state, queueing_network.node_edge_adjacency, estimated_cost_matrix, queueing_network.edge_capacities, nu)
        
        # take action and update the network state 
        network_status = queueing_network.step(planned_edge_rates)

    return queueing_network


def calculate_costs(queueing_network):
    tran_cost_per_time_per_run = queueing_network.planned_edge_rates.transpose((0,2,1))@queueing_network.true_edge_costs
    tran_cost_till_tt = np.mean(np.cumsum(tran_cost_per_time_per_run, axis=1), axis=0)
    tran_cost_at_tt = np.mean(tran_cost_per_time_per_run, axis=0)

    nonull_tran_cost_per_time_per_run = queueing_network.actual_edge_rates.transpose((0,2,1))@queueing_network.true_edge_costs
    nonull_tran_cost_till_tt = np.mean(np.cumsum(nonull_tran_cost_per_time_per_run, axis=1), axis=0)
    nonull_tran_cost_at_tt = np.mean(nonull_tran_cost_per_time_per_run, axis=0)

    backlog_at_tt = np.mean(np.sum(queueing_network.queues, axis=1), axis=0)[1:]
    backlog_cost_at_tt = backlog_at_tt*np.sum(queueing_network.true_edge_costs)

    return tran_cost_till_tt, nonull_tran_cost_till_tt, tran_cost_at_tt, nonull_tran_cost_at_tt, backlog_cost_at_tt, backlog_at_tt


