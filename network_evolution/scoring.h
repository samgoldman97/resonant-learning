#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include "dict.h"
#include "helpers.h"



#define T_MAX_MULTIPLE 35
// Hub behavior enum moved to helpers 


/* 
 * Helper function to abstract the behavior of oscillation
 * behavior: enum flag to determine behavior
 * current_state: Pointer to the hub node to be changed
 * time: Current time for oscillation
 * start_state: initial value (important for oscillation and block)
 * period: period of oscillation
 * stereotyped_seq: sequence of stereotyped noise of length period.. 
 */
void modify_state(hub_behavior behavior, bool* current_state, int time, int start_state, int period, int* stereotyped_seq);

/* 
 * Score the function for match! 
 * target: Target function
 * cycle: array containing the cycle
 * cycle_length: Length of cycle
 * target_length: Length of target
 * See paper for details
 */
double score_cycle(int* target, int* cycle, int target_length, int cycle_length);


/* 
 * c_oscillation_score
 * Find a cycle with the given period and oscillation
 * Return the score as per scoring function
 * net: pointer to network datatype 
 * state: 1d array of state
 * n: num nodes
 * time_steps: max time steps
 * node_index: Index of hub node to oscillation
 * period: period to oscillate node at
 * target_fn: target sequence
 * target_length: Length of the target function
 * target_node: Node to take the target of
 * behavior: Flag for behavior
 * stereotyped_seq: int* for sequence of length period if we have a sterotyped behavior
 * check_mini_cycles: int flag.  If != 0 , look for attractors not only at intervals of the period, 
 *  but at every occurence
 * include_input: int flag. If == 0, (false), don't include the input when considering attractors of the network
 *  To do this, change the hash.. 
 */
double c_score_net (network* net, int* state, int n,
                    int time_steps, int node_index, int period, int* target_fn, int target_length, 
                    int target_node, hub_behavior behavior, int* stereotyped_seq, int check_mini_cycles, 
                    int include_input); 

/* 
 * c_update
 * Function to update a state for a maximum number of time steps before hitting attractor
 * net: pointer to network datatype 
 * state: 1d array of state
 * network_states: 2d array to hold the output of this table!
 * n: num nodes
 * time_steps: max time steps
 * node_index: Index of hub node to oscillation.. Set to 0 for non-oscil
 * period: period to oscillate node at... Set to 1 for non-oscil
 * behavior: flag for hub node behavir
 * include_input: int flag. If == 0, (false), don't include the input when considering attractors of the network
 *  To do this, change the hash.. 
 */
void c_update (network* net, int* state, int* network_states, 
               int n, int time_steps, int node_index, int period, hub_behavior behavior,
               int include_input); 

/* 
 * c_evolution_multiplex 
 * array: Adjacency table to convert 
 * n: Number of nodes
 * time_steps: Max number of time steps to try updating each network 
 * generations: number of generations
 * node_indices: hub nodes to oscillate in each passed network
 * periods: periodicities to oscillate the hub node
 * target_lengths: lengths of target functions
 * target_node: target node to be oscillated. If -1, find a new one, else use
 * num_functions: How many different functions and periods we have..
 * behavior: enum flag to set the hub nodes behavior
 * average_scores: store the average of each score
 * population_size: number of networks in population -- should be 50!
 * num_replicates: number of child networks per network -- let's keep this at 3
 * selection_delay: Number of generations to average scores over before selection. 
    Use this flag for avging over multiplex. If selecting on each step, set to 1; every other, 2
 * sort_first: If true, sort the networks THEN append the top 50 scores. If false, append top 50 scores then sort
 * MSE: If 1, use mean squared error for selection, else use average error.
 * learning_blocks: number of time to repeat each target function in a row
 * learning_rotation: integer array of length generations for the index of the periods and target functions. This dictates the function learned at each generation
    If this is not null, then each time step learns a single function of target_functions whose index is given by learning_rotation[g]
 * record_all: pointer to a 2D array of floats.  We will monitor ALL the target functions during each update steps to get a better sense of how it changes over time
 * record_mutations: if true, save the mutations that accumulate
 * check_mini_cycles: int flag.  If == 0 (false) , look for attractors only at intervals of the period, 
 *  Else, look for cycles at every occurence
 * include_input: int flag. If == 0, (false), don't include the input when considering attractors of the network
 *  To do this, change the hash.. 
 * fix_start: int flag. If == 0, don't use set random starts. If it is 1, use set start states.. 
 * p_perturb: Probability of perturbing the fixed start state at each index.. 
 * swithc time: when if at all to switch targets
 */
void c_evolution_multiplex (double* array, int n, int time_steps, int generations,
                            int* node_indices, int* periods, int* target_lengths, int target_node, int num_functions,
                            hub_behavior behavior, double* average_scores, int population_size, int num_replicates,
                            int selection_delay, int sort_first, int MSE, int learning_blocks, int* learning_rotation,
                            double* record_all, int record_mutations, int check_mini_cycles, int include_input, int fix_start,
                            double p_perturb, int switch_time);

////////////////////////////////////////////////////////////
/// Update Wrapper functions
////////////////////////////////////////////////////////////
/* 
 * Axuilary function to go from python array to C adjacency list
 */
void c_update_wrapper(double*array, int* state, int* state_table, 
                      int n, int time_steps, int* r_seed, int include_input);

/* 
 * Axuilary function to go from python array to C adjacency list
 */
void c_update_oscillation_wrapper (double* array, int* state, int* state_table, 
                                   int n, int time_steps, int node_index, int period,
                                   int* r_seed, int include_input);

////////////////////////////////////////////////////////////
/// Evolution Wrapper functions
////////////////////////////////////////////////////////////

/* 
 *  Wrapper to call the evolution function with no hub node modificatoins
 */ 
void c_evolution_wrapper_free(double* array, int n, int time_steps, int generations,
                              double* scores, int pop_size, int num_replicates, int target_length, int target_node,
                              int sort_first, int record_mutations, int* r_seed, int check_mini_cycles, 
                              int include_input); 
/* 
 *  Wrapper to call the evolution function with random hub
 */
void c_evolution_wrapper_random(double* array, int n, int time_steps, int generations,
                                int* node_indices, double* scores, int pop_size, int num_replicates, 
                                int target_length, int target_node, int sort_first, int record_mutations, int* r_seed, 
                                int check_mini_cycles, int include_input); 

/* 
 *  Wrapper to call the evolution function with randomly, constant through generations, imposed sequences on the hub
 */ 
void c_evolution_wrapper_no_freq(double* array, int n, int time_steps, int generations,
                               int* node_indices, double* scores, int pop_size,
                               int num_replicates, int target_length, int target_node, int sort_first,
                               int record_mutations, int* r_seed, int check_mini_cycles,
                               int include_input); 

/*
 *  Wrapper to call the evolution function with hub oscillations
 */ 
void c_evolution_wrapper_oscil(double* array, int n, int time_steps, int generations,
                               int* node_indices, int period, double* scores, int pop_size,
                               int num_replicates, int target_length, int target_node, int sort_first,
                               int stereotyped, int record_mutations, int* r_seed, 
                               int check_mini_cycles, int include_input);


////////////////////////////////////////////////////////////
/// Evolution Multiplex Wrapper Functions
////////////////////////////////////////////////////////////

/* 
 *  Wrapper to call the evolution function with no hub node modifications
 */ 
void c_evolution_multiplex_wrapper_free(double* array, int n, int time_steps, int generations,
                                        double* scores, int pop_size, int num_replicates,
                                        int* target_lengths, int target_node, int num_functions, int selection_delay,
                                        int sort_first, int MSE, int learning_blocks,
                                        int* learning_rotation, double* record_all, int record_mutations,
                                        int* r_seed, int check_mini_cycles, int include_input); 


/* 
 *  Wrapper to call the evolution function with multiple targets and a randomly, constant through generations, imposed sequences on the hub
 */ 
void c_evolution_multiplex_wrapper_no_freq(double* array, int n, int time_steps, int generations,
                                         int* node_indices, double* scores, int pop_size, int num_replicates,
                                         int* target_lengths, int target_node, int num_functions, int selection_delay,
                                         int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, double* record_all, int record_mutations,
                                         int* r_seed, int check_mini_cycles, int include_input); 



/* 
 *  Wrapper to call the evolution function with multiple targets and hub oscillations
 */ 
void c_evolution_multiplex_wrapper_oscil(double* array, int n, int time_steps, int generations,
                                         int* node_indices, int* periods, double* scores, int pop_size,
                                         int num_replicates, int* target_lengths, int target_node, int num_functions,
                                         int selection_delay, int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, int stereotyped, double* record_all,
                                         int record_mutations, int* r_seed, int check_mini_cycles, 
                                         int include_input, int switch_time); 



/* 
 *  Wrapper to call the evolution function with multiple targets and hub oscillations
 */ 
void c_evolution_multiplex_fixed_starts(double* array, int n, int time_steps, int generations,
                                         int* node_indices, double* scores, int pop_size, int num_replicates,
                                         int* target_lengths, int target_node, int num_functions, int selection_delay,
                                         int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, double* record_all, int record_mutations,
                                         int* r_seed, int include_input, double p, int constant_hub); 
