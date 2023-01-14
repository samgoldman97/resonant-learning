/*
scoring.c

simple C function that alters data passed in via a pointer

    used to see how we can do this with Cython/numpy

*/

#include "scoring.h"


/* 
 * Helper function to abstract the behavior of oscillation
 * behavior: enum flag to determine behavior
 * current_state: Pointer to the hub node to be changed
 * time: Current time for oscillation
 * start_state: initial value (important for oscillation and block)
 * period: period of oscillation
 * stereotyped_seq: sequence of stereotyped noise of length period if streotyped behavior.  If NO_FREQ, this should have length >= time.. 
 */
void modify_state(hub_behavior behavior, bool* current_state, int time, int start_state, int period, int* stereotyped_seq){
    int half_period; 
    int phase; 
    switch (behavior){
        case FREE: 
            break; 
        case OSCILLATE:
            half_period = period / 2; 
            phase = time % period; 
            current_state[0] = (start_state + (phase >= half_period)) % 2; 
            break; 
        case BLOCK_ON: 
            current_state[0] = 1; 
            break;
        case BLOCK_OFF: 
            current_state[0] = 0; 
            break; 
        case STEREOTYPED:
            current_state[0] = stereotyped_seq[time % period];
            break;
        case RANDOM:
            current_state[0] = rand_bool(); 
            break;
        case NO_FREQ: 
            current_state[0] = stereotyped_seq[time]; 
            break;
    }
    
    return; 

}

/* 
 * Score the function for match! 
 * target: Target function
 * cycle: array containing the cycle
 * cycle_length: Length of cycle
 * target_length: Length of target
 * See paper for details
 */
double score_cycle(int* target, int* cycle, int target_length, int cycle_length){
    
    int lowest_denom = lcm(cycle_length, target_length); // cycle_length * target_length; 
    int max_val = T_MAX_MULTIPLE * cycle_length; 
    int sum_bound = (lowest_denom > max_val) ? max_val : lowest_denom; 

    double cur_score; 
    double max_score = 0;

    // Loop over all permutations of the target!
    for (int l = 0; l < target_length; l++){
        cur_score = 0; 
        // Calculate sum 
        for (int j = 0; j < sum_bound; j++){
            // printf("Target: %d, Cycle: %d\n", target[(j + l) % target_length], cycle[j % cycle_length]); 
            cur_score += abs(target[(j + l) % target_length] - cycle[j % cycle_length]);
        }

        // Convert to average
        cur_score /= sum_bound;
        cur_score = 1 - cur_score;

        // Reset max value
        if (cur_score > max_score){
            max_score = cur_score; 
        }
    }

    // printf("Cycle length: %d; Score: %.2f; ", cycle_length, max_score); 
    // for (int j = 0; j < cycle_length; j++){
    //     printf("%d ", cycle[j]);
    // }
    // printf("\n");
    return max_score; 

 }

/* 
 * c_score_net
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
                    int target_node, hub_behavior behavior, int* stereotyped_seq, 
                    int check_mini_cycles, int include_input) {
    // printf("START SCORE NET\n");
    hashtable_t* my_tbl = ht_create(HASH_SIZE,n);

    entry_t* my_entry; 
    uint32_t bin; 
    int hash_result; 

    // Guarantee oscillatory node is low
    int start_state = state[node_index]; 
    // For free behavior
    if (behavior == FREE){
        period = 1; 
        node_index = 0; 
    }

    // Create an array to hold pointers to all the network states
    // Keep this on the stack
    int target_states[time_steps]; 
    // Default if we don't find an attractor... 
    double cycle_score = -1; 

    // Declare states... 
    bool* current_state; 
    bool* previous_state = malloc(sizeof(bool) * n); 

    // Set previous_state[0:n] = state, the initial conditions
    double running_sum[n]; 

    // K = 1 Rule 
    // Temp
    // int num_inputs[n];
    // int only_input[n];

    for (int j = 0; j < n; j++){
        previous_state[j] = (bool) state[j]; 
        running_sum[j] = 0; 
        // K = 1 Rule
        // num_inputs[j] = 0;
    }

    node * cur_node; 
    // printf("PRE TIME STEP\n"); 
    // Loop over number of time steps
    for (int t = 0; t < time_steps; t++){
        // Modify the hub input node... 
        modify_state(behavior, &previous_state[node_index], t, start_state, period, stereotyped_seq);
        bin = FNV_OFFSET_32; 
        current_state = malloc(sizeof(bool) * n); 

        // Actual update loop
        for (int i = 0; i < n; i++) {
            cur_node = net->nodes[i];
            while (cur_node){
                // OUT DEGREE LINKED LIST 
                running_sum[cur_node->target_node] += cur_node->weight * previous_state[i]; 
                // IN DEGREE LINKED LIST 
                // running_sum[i] += cur_node->weight * previous_state[cur_node->target_node]; 

                // K = 1
                // OUT DEGREE LINKED LIST 
                // num_inputs[cur_node->target_node] += 1; 
                // only_input[cur_node->target_node] = i; 
                // IN DEGREE LINKED LIST 
                // num_inputs[i] += 1; 
                // only_input[i] = cur_node->target_node; 

                cur_node = cur_node->next; 
            }
        }

        bin = FNV_OFFSET_32; 
        for (int i = 0; i < n; i ++){
            // Update state i without branching using threshold rule
            // K = 1
            // TEMP
            // if (num_inputs[i] == 1){
            //     current_state[i] = (running_sum[i] > 0) * previous_state[only_input[i]] - (running_sum[i] < 0) * previous_state[only_input[i]];
            // } else { 
            current_state[i] = (running_sum[i] > 0) + (running_sum[i] == 0) * previous_state[i];
            // }   
            // Handle hashing in the same lopo
            bin =  bin ^ previous_state[i]; // xor next byte into the bottom of the hash
            bin = bin * FNV_PRIME_32;
            running_sum[i] = 0; 

            // K = 1
            // TEMP
            // num_inputs[i] = 0; 
        }  


        // Set the output!
        target_states[t] = previous_state[target_node]; 
        // instead of looping again, calculate the hash as we go.. 
        // bin = FNV32(previous_state, n); 
        // If check mini cycles, we will record this state on EVERY time step, 
        // not just multiples of period
        if ((t % period == 0) || check_mini_cycles){
            my_entry = malloc(sizeof(entry_t)); 
            my_entry->key = previous_state; 
            my_entry->value = t; 
            my_entry->next = NULL; 

            // If we don't want to include the input in the calculation, 
            // Then just set the node_index for the input to 0, adefault
            if (include_input == 0){
                previous_state[node_index] = 0; 
            }

            bin = bin % (HASH_SIZE- 1); 
            hash_result = ht_check_insert(my_tbl, my_entry, bin); 

            // If we found a hit... 
            if (hash_result != -1){
                // printf("Cycle Length: %d\n",t - hash_result); 
                cycle_score =  score_cycle(target_fn, &target_states[hash_result], target_length, t - hash_result);

                // printf("cycle length: %d\n", t - hash_result);
                // Was not inserted; free
                free(previous_state); 
                // printf("Cycle score: %.2f\n", cycle_score);
                break; 
            }
        } else {
            free(previous_state); 
        }
        previous_state = current_state; 
    }
    free(current_state); 
    free_hash_table(my_tbl);
    if (cycle_score == -1){
      printf("Did not find cycle!\n");
      cycle_score = 0; 
    }
    return cycle_score; 
}

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
               int include_input){

    hashtable_t* my_tbl = ht_create(HASH_SIZE,n);
    entry_t* my_entry; 
    uint32_t bin; 
    int hash_result; 

    // Guarantee oscillatory node is low
    int start_state = state[node_index]; 

    int* stereotyped_seq = NULL;
    // For free behavior
    if (behavior == FREE){
        period = 1; 
        node_index = 0; 
    } else if (behavior == STEREOTYPED){
        stereotyped_seq = rand_target(period); 
    }
    // Declare states... 
    bool* current_state; 
    bool* previous_state = malloc(sizeof(bool) * n); 

    // Set previous_state[0:n] = state, the initial conditions
    double running_sum[n]; 
    for (int j = 0; j < n; j++){
        previous_state[j] = (bool) state[j]; 
        running_sum[j] = 0; 
    }

    node * cur_node; 
    // Loop over number of time steps
    for (int t = 0; t < time_steps; t++){
        modify_state(behavior, &previous_state[node_index], t, start_state, period, stereotyped_seq);

        current_state = malloc(sizeof(bool) * n); 

        // Actual update loop
        for (int i = 0; i < n; i++) {
            cur_node = net->nodes[i];
            while (cur_node){
                // OUT degree implementation 
                running_sum[cur_node->target_node] += cur_node->weight * previous_state[i]; 
                
                // In degree implementation 
                // running_sum[i] += cur_node->weight * previous_state[cur_node->target_node]; 

                cur_node = cur_node->next; 
            }

        }

        bin = FNV_OFFSET_32; 
        // update all nodes... 
        for (int i = 0; i < n; i ++){
            // Update state i without branching using threshold rule
            current_state[i] = (running_sum[i] > 0) + (running_sum[i] == 0) * previous_state[i];
                        
            // Set the output!
            network_states[t * n + i] = previous_state[i];

            // Handle hashing in the same oop
            bin =  bin ^ previous_state[i]; // xor next byte into the bottom of the hash
            bin = bin * FNV_PRIME_32;
            running_sum[i] = 0; 
        }  


        // instead of looping again, calculate the hash as we go.. 
        // bin = FNV32(previous_state, n); 
        if (t % period == 0){
            bin = bin % (HASH_SIZE- 1); 
            my_entry = malloc(sizeof(entry_t)); 
            my_entry->key = previous_state; 
            my_entry->value = t; 
            my_entry->next = NULL; 

            // If include input is false, set input to 0... 
            if (include_input == 0){
                previous_state[node_index] = 0; 
            }

            hash_result = ht_check_insert(my_tbl, my_entry, bin); 

            if (hash_result != -1){
                // printf("Cycle started at time: %d\n", hash_result);
                // printf("Cycle ended at time: %d\n", t);  
                // Was not inserted, free it!
                free(previous_state); 
                break; 
            }
        } else {
            // Free the previous state since we don't need it anymore... 
            free(previous_state); 
        }
        previous_state = current_state; 
    }

    // Write output back to the state array
    // Let's take previous state...
    for (int j = 0; j < n; j++){
        state[j] = (int) previous_state[j]; 
    }
    free(current_state); 
    free_hash_table(my_tbl);
    if (stereotyped_seq){
        free(stereotyped_seq); 
    }
}

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
 * switch_time: If -1, don't switch, else reroll the dice for the target sequences 
 */
void c_evolution_multiplex (double* array, int n, int time_steps, int generations,
                            int* node_indices, int* periods, int* target_lengths, int target_node, int num_functions,
                            hub_behavior behavior, double* average_scores, int population_size, int num_replicates,
                            int selection_delay, int sort_first, int MSE, int learning_blocks, int* learning_rotation,
                            double* record_all, int record_mutations, int check_mini_cycles, int include_input, int fix_start,
                            double p_perturb, int switch_time){

    int* target_fns[num_functions];
    int* stereotyped_seqs[num_functions]; 
    // If we want to fix the start 
    int* input_states[num_functions]; 

    for (int i = 0; i < num_functions; i++){
        target_fns[i] = rand_target(target_lengths[i]);
        stereotyped_seqs[i] = NULL; 
        if (behavior == STEREOTYPED){
            stereotyped_seqs[i] = rand_target(periods[i]); 
        } else if (behavior == NO_FREQ){
            // If we need a stereotyped input noise, generate a function as long as the time 
            stereotyped_seqs[i] = rand_target(time_steps); 
        }
        // If we want to hold # of network start states 
        if (fix_start){
            input_states[i] = rand_network_state(n);
        }
    }

    int fn_index; 
    int target_length; 
    int* target_fn;
    int* stereotyped_seq; 
    int period;

    // Use these if we want to keep track of block learning and unlearning
    int target_length_temp;
    int* target_fn_temp;
    int* stereotyped_seq_temp;
    int period_temp; 
    
    double temp_score; 
    int population_capacity = population_size * (1 + num_replicates);  

    // Stable 
    int stable_gens = 100; 
    int current_stable = 0; 



    // Only update if target node is not set.. 
    if (target_node == -1){
        target_node = rand_int_range(0, n); 
    }

    // Ensure we don't oscillate our target
    while (node_indices && linear_search(node_indices,population_size,target_node)
            && behavior != FREE){
        target_node = rand_int_range(0, n); 
    }

    // If we are recording mutations, make an array to hold all the muation first nodes..
    mutation_list** mutation_recordings = NULL; 
    if (record_mutations){
        mutation_recordings = malloc(sizeof(mutation_list) * generations);
        for (int r = 0; r < generations; r++){
            mutation_recordings[r] = NULL; 
        }
    }
    
    // Step one, convert array

    // Step two: create population size array 
    network** population = malloc(sizeof(network*) * population_capacity); 
    // Scores will hold the score of the networks at each time step
    // Avg scores will hold the avg over the selection delay steps
    double* scores = malloc(sizeof(double) * population_capacity);
    double* avg_scores = malloc(sizeof(double) * population_capacity);
    double prev_avg = 0; 

    for (int i = 0; i < population_capacity; i++){
        population[i] = NULL; 
        scores[i] = 0; 
        avg_scores[i] = 0; 
    }

    // int num_networks = population_size;     
    for (int i =0; i < population_size; i++){
        population[i] = convert_mat(&array[i * n * n], n); 
        if (node_indices){
            population[i]->oscil_node = node_indices[i];
        }
    }

    int g; 
    // For g generations!!!
    for (g = 0; g < generations; g++){
        // printf("Generation %d\n", g);


        // If we need to switch target functions: 
        if (g == switch_time){
            for (int i = 0; i < num_functions; i++){
                free(target_fns[i]); 
                target_fns[i] = rand_target(target_lengths[i]);
            }
        }



        if (!learning_rotation){
            fn_index = (g / learning_blocks) % num_functions; 
        } else {
            fn_index = learning_rotation[g]; 
        }

        target_length = target_lengths[fn_index];
        target_fn = target_fns[fn_index]; 
        period = periods ? periods[fn_index] : 1;
        stereotyped_seq = (behavior == STEREOTYPED || behavior == NO_FREQ) ? stereotyped_seqs[fn_index] : NULL; 

        // Step two: generate diversity
        // If we are ready to generation diversity
        // If we sort first, then we want to do one trial with no diversity as a baseline
        // If we are sorting after recording the score, we can still generate diversity on the first attempt
        // Otherwise, generating diversity on the first step will lead to never recording the scores on the first step
        if ( (g > 0 || !sort_first)  && g % selection_delay == 0){
            // Before resetting the score, hold onto it for output.. 
            prev_avg = avg_scores[0];
            for (int i = 0; i < population_size; i++){
                // reset score!
                if (population[i]){
                    avg_scores[i] = 0; 
                    scores[i] = 0; 
                    for (int j =0; j < num_replicates; j++){
                        population[i + ((j+1) * population_size)] = mutate_network(population[i], record_mutations); 
                        avg_scores[i + ((j+1) * population_size)] = 0; 
                        scores[i + ((j+1) * population_size)] = 0; 

                    }    
                    // printf("Found a non null network\n"); 
                }
            }
        }
        
        // Step three: generate scores 
        // Question: Should I have a random start on each network for each generation?
        // OR should we just have one random start per generaiton (more noise)
        //      num_networks = 0; 
        for (int i = 0; i < population_capacity; i++){

            int* random_start; 
            if (fix_start){
                random_start = copy_network_state(n, input_states[fn_index]); 
                // Perturb all nodes with probability p_perturb
                if (p_perturb > 0){
                    perturb_start(n, random_start, p_perturb); 
                }
            } else {
                random_start = rand_network_state(n);
            }
            if (population[i]){
                temp_score = c_score_net(population[i], random_start, n, time_steps,
                                         population[i]->oscil_node, period, target_fn,
                                         target_length, target_node, behavior, stereotyped_seq,
                                         check_mini_cycles, include_input); 
                if (MSE){ 
                    // Mean squared error... 
                    avg_scores[i] +=  temp_score * temp_score  / selection_delay; 
                } else {
                    avg_scores[i] += temp_score / selection_delay;
                }

                scores[i] = temp_score; 

                // num_networks++; 
            }
            free(random_start);


            // This is a hack... If we want to get all the scores at each step... 
            // Usually, the averaging of each function is abstracted away into the score_net function..
            if (record_all){
                for (int temp_index = 0; temp_index < num_functions; temp_index++){

                    target_length_temp = target_lengths[temp_index];
                    target_fn_temp = target_fns[temp_index]; 
                    period_temp = periods ? periods[temp_index] : 1;
                    stereotyped_seq_temp = (behavior == STEREOTYPED || behavior == NO_FREQ) ? stereotyped_seqs[temp_index] : NULL;
                    
                    int* random_start; 
                    if (fix_start){
                        random_start = copy_network_state(n, input_states[temp_index]);
                        // Perturb all nodes with probability p_perturb
                        perturb_start(n, random_start, p_perturb); 
                    } else {
                        random_start = rand_network_state(n);
                    }
                    if (population[i] && i < population_size){
                        record_all[g*num_functions + temp_index] += c_score_net(population[i], random_start, n, time_steps,
                                                                    population[i]->oscil_node, period_temp, target_fn_temp,
                                                                    target_length_temp, target_node, behavior,
                                                                    stereotyped_seq_temp, check_mini_cycles, include_input) / population_size; 

                    }
                    free(random_start);
                }
            }
        }
        // Step 3.. we need to get winners... 
        // num_networks = num_networks > population_size ? population_size : num_networks; 
        
        // Sort and add scores only if we are one generation right before doing new mutations
        // Sort based on avg scores, but append temp scores to the list! We can avg scores after
        // Do not sort until we are at 
        if ( (g  % selection_delay == (selection_delay - 1))  && sort_first){
            sort_winners(population_size, population_capacity, population, avg_scores); 
            average_scores[g] = calculate_average(scores, population_size); 
            // Check if we've finished
            if (fabs(1 - average_scores[g]) < 0.00001){
                current_stable += 1; 
            } else {
                current_stable = 0; 
            }
        } else if ( (g  % selection_delay == (selection_delay - 1)) && !sort_first) {
            average_scores[g] = calculate_average(scores, population_size); 
            sort_winners(population_size, population_capacity, population, avg_scores); 
            // Check if we've finished... 
            if (fabs(1 - average_scores[g]) < 0.00001){
                current_stable += 1; 
            } else {
                current_stable = 0; 
            }
        } else { 
            // If we are not at a sorting stage, just add the score to the list of scores..
            average_scores[g] = calculate_average(scores,population_size);
        }

        // record mutations if we just sorted...
        // Only record for population[0] the highest scorer... 
        if (mutation_recordings && (g % selection_delay == (selection_delay - 1))){
            copy_mutation_list(population[0]->mutation_list, &mutation_recordings[g]); 
        }

        
        // Free losers if we just sorted
        if ( g % selection_delay == (selection_delay - 1) ){
            for (int j = population_size; j < population_capacity; j++){
                if (population[j]){
                    free_network(population[j], n);
                    population[j] = NULL;
                }
            }
        }

        if (current_stable > stable_gens){
            // printf("BROKEN AT GEN %d\n", g); 
            // If we're waiting on switch time, don't break, just send forward to the switch time
            if (g < switch_time){
                // printf("bumping g ahead!!\n")\n
                // printf("Confirming broken: gen %d\n", g); 
                for (g += 0; g < switch_time; g++){
                    average_scores[g] = 1; 
                }
                g = switch_time - 1;
                current_stable = 0; 
 

            } else {
                break; 
            }
        }
    }


    // printf("Confirming broken: gen %d\n", g); 
    for (g += 0; g < generations; g++){
        average_scores[g] = 1; 
    }
    //////  OUTPUTTING 

    // Convert networks to output! 
    for (int i =0 ; i < population_size; i++){
        net_to_mat(population[i], &array[i * n * n]); 
    }

    write_out_files("Multiplex_Evolution", population[0], target_fns, target_lengths,
                    num_functions, target_node, generations,
                    population_size, selection_delay, sort_first, 
                    behavior, prev_avg, MSE, periods, stereotyped_seqs); 

    // Note: this frees the mutations inside the function!

    if (mutation_recordings){
        write_out_mutations("Mutations", generations, mutation_recordings);
    }

    // step 6... Free networks and scores
    for (int i = 0; i < population_capacity; i++){
        if (population[i]){
            free_network(population[i], n); 
        }
    }
    for (int i = 0; i < num_functions; i ++){
        free(target_fns[i]);
        if (stereotyped_seqs[i]){
            free(stereotyped_seqs[i]); 
        }

        if (fix_start){
            free(input_states[i]); 
        }

    }

    free(population); 
    free(scores); 
    free(avg_scores); 
}

////////////////////////////////////////////////////////////
/// Update Wrapper functions
////////////////////////////////////////////////////////////
/* 
 * Axuilary function to go from python array to C adjacency list
 */
void c_update_wrapper(double*array, int* state, int* state_table, 
                      int n, int time_steps, int* r_seed, int include_input){

    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    
    network* net = convert_mat(array, n); 
    hub_behavior behavior = FREE; 
    int node_index = 0; 
    int period = 1; 
    c_update(net, state, state_table, n, time_steps, 
        node_index, period, behavior, include_input); 
    free_network(net, n); 
}

/* 
 * Axuilary function to go from python array to C adjacency list
 */
void c_update_oscillation_wrapper (double* array, int* state, int* state_table, 
                                   int n, int time_steps, int node_index, int period,
                                   int* r_seed, int include_input) {
    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    network* net = convert_mat(array, n); 
    hub_behavior behavior = OSCILLATE; 
    c_update(net, state, state_table, n, time_steps, 
        node_index, period, behavior, include_input); 
    free_network(net, n); 
}

////////////////////////////////////////////////////////////
/// Evolution Wrapper functions
////////////////////////////////////////////////////////////

/* 
 *  Wrapper to call the evolution function with no hub node modificatoins
 */ 
void c_evolution_wrapper_free(double* array, int n, int time_steps, int generations,
                              double* scores, int pop_size, int num_replicates, int target_length,
                              int target_node, int sort_first, int record_mutations, int* r_seed, int check_mini_cycles, 
                              int include_input){


    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }    

    hub_behavior behavior = FREE; 
    int num_functions = 1; 
    int selection_delay = 1; 
    int MSE = 0; 
    int learning_blocks = 1; 
    // int record_mutations = 0; 

    // Call multiplex instead of evolution...
    c_evolution_multiplex(array, n, time_steps, generations,
                          NULL, NULL, &target_length, target_node, num_functions, behavior, scores, 
                          pop_size, num_replicates, selection_delay, sort_first, MSE, 
                          learning_blocks, NULL, NULL, record_mutations, check_mini_cycles, 
                          include_input, 0, 0, -1); 

    // c_evolution(array, n, time_steps,generations, 
    //             NULL, 1, behavior, scores, pop_size,
    //             num_replicates, target_length, sort_first); 

}


/* 
 *  Wrapper to call the evolution function with random hub
 */
void c_evolution_wrapper_random(double* array, int n, int time_steps, int generations,
                                int* node_indices, double* scores, int pop_size, int num_replicates,
                                int target_length, int target_node, int sort_first, int record_mutations, int* r_seed,
                                int check_mini_cycles, int include_input){

    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    hub_behavior behavior = RANDOM;
    int num_functions = 1; 
    int selection_delay = 1; 
    int MSE = 0; 
    int learning_blocks = 1; 
    // int record_mutations = 0; 
        
    // Call multiplex instead of evolution.. .
    c_evolution_multiplex(array, n, time_steps, generations,
                          &node_indices[0], NULL, &target_length, target_node,
                          num_functions, behavior, scores, pop_size, num_replicates,
                          selection_delay, sort_first, MSE, learning_blocks, NULL,
                           NULL, record_mutations, check_mini_cycles, include_input, 0, 0, -1); 

    // c_evolution(array, n, time_steps,generations, &node_indices[0],
    //             1, behavior, scores, pop_size, num_replicates,
    //             target_length, sort_first); 
}



/* 
 *  Wrapper to call the evolution function with randomly, constant through generations, imposed sequences on the hub
 */ 
void c_evolution_wrapper_no_freq(double* array, int n, int time_steps, int generations,
                               int* node_indices, double* scores, int pop_size,
                               int num_replicates, int target_length, int target_node, int sort_first,
                               int record_mutations, int* r_seed, int check_mini_cycles, int include_input){
    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    
    hub_behavior behavior = NO_FREQ; 
    int num_functions = 1; 
    int selection_delay = 1; 
    int MSE = 0; 
    int learning_blocks = 1; 
    int period = 1; 
        
    // Call multiplex instead of evolution.. .
    c_evolution_multiplex(array, n, time_steps, generations,
                          &node_indices[0], &period, &target_length, target_node, num_functions, behavior, scores, 
                          pop_size, num_replicates, selection_delay, sort_first, MSE, 
                          learning_blocks, NULL, NULL, record_mutations, check_mini_cycles, include_input, 0, 0, -1); 
}


/* 
 *  Wrapper to call the evolution function with hub oscillations
 */ 
void c_evolution_wrapper_oscil(double* array, int n, int time_steps, int generations,
                               int* node_indices, int period, double* scores, int pop_size,
                               int num_replicates, int target_length, int target_node, int sort_first,
                               int stereotyped, int record_mutations, int* r_seed, int check_mini_cycles,
                               int include_input){
    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    
    hub_behavior behavior = stereotyped ? STEREOTYPED : OSCILLATE; 
    int num_functions = 1; 
    int selection_delay = 1; 
    int MSE = 0; 
    int learning_blocks = 1; 
    // int record_mutations = 0; 
        
    // Call multiplex instead of evolution.. .
    c_evolution_multiplex(array, n, time_steps, generations,
                          &node_indices[0], &period, &target_length, target_node, num_functions, behavior, scores, 
                          pop_size, num_replicates, selection_delay, sort_first, MSE, 
                          learning_blocks, NULL, NULL, record_mutations, check_mini_cycles, include_input, 0,0, -1); 


    // c_evolution(array, n, time_steps,generations, &node_indices[0], 
    //             period, behavior, scores, pop_size, num_replicates,
    //             target_length, sort_first); 

}



////////////////////////////////////////////////////////////
/// Evolution Multiplex Wrapper Functions
////////////////////////////////////////////////////////////

/* 
 *  Wrapper to call the evolution function with no hub node modifications
 */ 
void c_evolution_multiplex_wrapper_free(double* array, int n, int time_steps, int generations,
                                        double* scores, int pop_size, int num_replicates,  int* target_lengths,
                                        int target_node, int num_functions, int selection_delay, int sort_first, int MSE,
                                        int learning_blocks, int* learning_rotation, double* record_all, int record_mutations,
                                        int* r_seed, int check_mini_cycles, int include_input){
    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    hub_behavior behavior = FREE;
    c_evolution_multiplex(array, n, time_steps,generations,
                          NULL, NULL, &target_lengths[0], target_node, num_functions, behavior, scores, pop_size,
                          num_replicates, selection_delay, sort_first, MSE, learning_blocks,
                          learning_rotation, record_all, record_mutations, check_mini_cycles, 
                          include_input, 0, 0, -1); 
}


/* 
 *  Wrapper to call the evolution function with multiple targets and a randomly, constant through generations, imposed sequences on the hub
 */ 
void c_evolution_multiplex_wrapper_no_freq(double* array, int n, int time_steps, int generations,
                                         int* node_indices, double* scores, int pop_size, int num_replicates,
                                         int* target_lengths, int target_node, int num_functions, int selection_delay,
                                         int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, double* record_all, int record_mutations,
                                         int* r_seed, int check_mini_cycles, int include_input){
    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    
    // Hub beahvior.. 
    hub_behavior behavior = NO_FREQ;
    int periods[num_functions];
    // For the multiplex function, we need periods of 1 so we find the earliest attractor cycle
    for (int i = 0; i < num_functions; i++){
        periods[i] = 1; 
    }
    c_evolution_multiplex(array, n, time_steps,generations, &node_indices[0],
                          &periods[0], target_lengths, target_node, num_functions, behavior, scores, pop_size,
                          num_replicates, selection_delay, sort_first, MSE, learning_blocks,
                          learning_rotation, record_all, record_mutations, check_mini_cycles, 
                          include_input, 0, 0, -1); 
}


/* 
 *  Wrapper to call the evolution function with multiple targets and hub oscillations
 */ 
void c_evolution_multiplex_wrapper_oscil(double* array, int n, int time_steps, int generations,
                                         int* node_indices, int* periods, double* scores, int pop_size, int num_replicates,
                                         int* target_lengths, int target_node, int num_functions, int selection_delay,
                                         int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, int stereotyped, double* record_all, int record_mutations,
                                         int* r_seed, int check_mini_cycles, int include_input, int switch_time){
    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    
    hub_behavior behavior = stereotyped ? STEREOTYPED : OSCILLATE;
    c_evolution_multiplex(array, n, time_steps,generations, &node_indices[0],
                          &periods[0], target_lengths, target_node, num_functions, behavior, scores, pop_size,
                          num_replicates, selection_delay, sort_first, MSE, learning_blocks,
                          learning_rotation, record_all, record_mutations, check_mini_cycles, include_input, 0, 0, switch_time); 
}

/* 
 *  Wrapper to call the evolution function with multiple targets and hub oscillations
 */ 
void c_evolution_multiplex_fixed_starts(double* array, int n, int time_steps, int generations,
                                         int* node_indices, double* scores, int pop_size, int num_replicates,
                                         int* target_lengths, int target_node, int num_functions, int selection_delay,
                                         int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, double* record_all, int record_mutations,
                                         int* r_seed, int include_input, double p, int constant_hub){

    if (r_seed){
        srand(r_seed[0]); 
    } else {
        int seed = time(NULL);
        srand(seed);
    }
    // Node indices will be 1
    hub_behavior behavior; 
    if (constant_hub > 0) {
        behavior = BLOCK_ON;
    } else {
        behavior = FREE; 
    }     
    c_evolution_multiplex(array, n, time_steps,generations, &node_indices[0],
                          NULL, target_lengths, target_node, num_functions, behavior, scores, pop_size,
                          num_replicates, selection_delay, sort_first, MSE, learning_blocks,
                          learning_rotation, record_all, record_mutations, 1, include_input, 1, p, -1); 
}



