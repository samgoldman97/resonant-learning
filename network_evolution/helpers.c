/* 
 * helpers.c
 * Helper functions for handling networks and generating random nums
 */ 

#include "helpers.h"


/* 
 * Convert net to matrix and store in mat 
 */
void net_to_mat(network* net, double* mat){

    int n = net->num_nodes;
    for (int i =0; i < n; i++){
        for (int j = 0; j < n; j++){
            mat[i*n + j] = 0;
        }
    }

    node* cur_node_old; 
    int j; 
    for (int i = 0; i < n; i++){
        cur_node_old = net->nodes[i]; 
        while (cur_node_old){

            j = cur_node_old->target_node; 
            // IN DEGREE LINKED LIST 
            // mat[i * n + j] = cur_node_old->weight; 
            // OUT DEGREE LINKED LIST
            mat[j * n + i] = cur_node_old->weight; 

            cur_node_old=cur_node_old->next; 
        }
    } 
}

/* 
 * Convert matrix into adjacency list... 
 * NOTE: THis is an adjacency list of the OUTGOING connections
 */ 
network* convert_mat(double* array, int n){
    // array is connections
    // printf("Starting conversion to adj list\n");
    network* my_net = malloc(sizeof(network)); 
    my_net->num_nodes = n; 
    my_net->nodes = (node**) malloc(sizeof(node*) * n); 
    my_net->num_outputs = malloc(n * sizeof(int));
    my_net->oscil_node = -1;
    my_net->mutation_list = NULL; 
    for (int i = 0; i < n; i++){
        my_net->nodes[i] = NULL;     
        my_net->num_outputs[i] = 0; 
    }
    int num_declared = 0;
    for (int j = 0; j < n; j++){
        node* cur_node =  my_net->nodes[j]; 
        for (int i = 0; i <n; i++){
            // if there's an input here
            // Add it to I
            // IN DEGREE LINKED LIST
            // if (array[j * n + i]  != 0){
            // OUT DEGREE LINKD LIST
            if (array[i * n + j]  != 0){
                num_declared++; 
                my_net->num_outputs[j]++; 
                if (cur_node){
                    cur_node->next = malloc(sizeof(node)); 
                    cur_node = cur_node->next; 
                } else {
                    my_net->nodes[j] =  malloc(sizeof(node));
                    cur_node = my_net->nodes[j]; 
                }
                cur_node->target_node = i;
                // OUT DEGREE LINKED LIST
                cur_node->weight = array[i * n + j]; 
                // IN DEGREE LINKED LIST
                // cur_node->weight = array[j * n + i]; 

                cur_node->next = NULL; 
            }
        }
    }
    // printf("In conversion, %d nodes declared\n", num_declared); 
    // printf("Ending conversion into adj matrix\n");
    return my_net; 
}


/* 
 * Copy a random state of a network
 */ 
int* copy_network_state(int n, int* net_state){
    int* state = malloc(sizeof(int) * n); 

    for (int i = 0; i < n; i++){
        state[i] = net_state[i]; 
    }
    return state; 
}


/* 
 * Deep copy a network.. 
 */ 
network* copy_network(network* old_net){
    network* new_net = malloc(sizeof(network)); 
    int n = old_net->num_nodes;
    new_net->num_nodes = n;
    new_net->nodes = (node**) malloc(sizeof(node*) * n);
    new_net->num_outputs = malloc(n * sizeof(int));
    new_net->oscil_node = old_net->oscil_node;
    new_net->mutation_list = NULL; 

    for (int i = 0; i < n; i++){
        new_net->nodes[i] = NULL;
        new_net->num_outputs[i] = old_net->num_outputs[i]; 
    }
    node* cur_node_old; 
    node* new_last;
    node* new_node; 
    for (int i = 0; i < n; i++){
        cur_node_old = old_net->nodes[i]; 
        new_last = NULL; 
        new_node = NULL; 
        while (cur_node_old){
            // Create new node

            new_node = malloc(sizeof(node)); 
            new_node->target_node = cur_node_old->target_node; 
            new_node->weight = cur_node_old->weight; 
            new_node->next = NULL; 

            // Add new node to linked list
            if (!new_last){
                new_net->nodes[i] = new_node; 
            } else{
                new_last->next = new_node; 
            }
            new_last = new_node; 
            cur_node_old=cur_node_old->next; 
        }
    } 
    return new_net; 
}


/* 
 * Perturb random state with probability p at each node
 */
void perturb_start(int n, int* net_state, double p){
    for (int i = 0 ; i < n; i++){
        if (rand_num() < p){
            net_state[i] = rand_bool();
        }
    }
}


/* 
 * Mutate a network!
 */ 
network* mutate_network(network* old_net, int record_changes){
    network* new_net = malloc(sizeof(network)); 
    int n = old_net->num_nodes;
    new_net->num_nodes = n;
    new_net->nodes = (node**) malloc(sizeof(node*) * n);
    new_net->num_outputs = malloc(n * sizeof(int));
    new_net->oscil_node = old_net->oscil_node;
    new_net->mutation_list = NULL; 
    for (int i = 0; i < n; i++){
        new_net->nodes[i] = NULL;
        new_net->num_outputs[i] = old_net->num_outputs[i]; 
    }

    node* cur_node_old; 
    node* new_last;
    node* new_node;
    // Type of mutatoin 
    mutation_type mutation;
    // Node to be mudated 
    node* mutation_node; 
    // How far in the linekd list 
    int depth_ctr; 
    mutation_list* cur_mutation = NULL; 
    
    // Index of node in the linked list to mutate
    int mutate_index = -1; 
    for (int i = 0; i < n; i++){

        cur_node_old = old_net->nodes[i]; 
        new_last = NULL; 
        new_node = NULL; 
        mutation_node = NULL; 
        mutation= NONE; 
        mutate_index = -1;
        depth_ctr = 0; 

        if ((rand_num() < MUTATION_THRESHOLD) && (new_net->num_outputs[i] > 0)){
            // Set mutation flag.. 
            // Make sure if we change an edge, that there's somewhere to change it to
            if (rand_bool() && new_net->num_outputs[i] < new_net->num_nodes){
                mutation = EDGE; 
            } else {
                mutation = WEIGHT; 
            }
            mutate_index = rand_int_range(0, new_net->num_outputs[i]);
            // Declare mutation list node and add to list
            if (record_changes){
                if (new_net->mutation_list && cur_mutation){
                    cur_mutation->next = malloc(sizeof(mutation_list));
                    cur_mutation = cur_mutation->next; 
                                       
                } else {
                    new_net->mutation_list = malloc(sizeof(mutation_list)); 
                    cur_mutation = new_net->mutation_list;                     
                }

                cur_mutation->mutate_node_src = i;
                cur_mutation->next=NULL;
            } 
        }
        
        while (cur_node_old){
            // Create new node
            new_node = malloc(sizeof(node)); 
            new_node->target_node = cur_node_old->target_node; 
            new_node->weight = cur_node_old->weight; 
            new_node->next = NULL; 

            // Check to see if this is something we should mutate
            if (depth_ctr == mutate_index){
                mutation_node = new_node; 
            } 


            // Add new node to linked list
            if (!new_last){
                new_net->nodes[i] = new_node; 
            } else{
                new_last->next = new_node; 
            }
            new_last = new_node; 
            cur_node_old=cur_node_old->next; 

            // Increment counter.. 
            depth_ctr++; 
        }

        // Handle mutations now!
        if (mutation == EDGE){
            int mutate_success = 0;
            int mutation_attempts = 0;
            int new_target; 
            int was_found; 
            int attempts_limit = 10; 
            // TODO: Change mutation attempts... 
            while (!mutate_success && mutation_attempts < attempts_limit ){
                new_target = rand_int_range(0, n); 
                // Make sure that this new random target 
                // Allow self loops
                was_found = 0; 
                node* search_node = new_net->nodes[i]; 
                while ((search_node) && (!was_found)){
                    was_found = (search_node->target_node == new_target);
                    search_node = search_node->next; 
                }

                if (was_found){
                    // If we can't handle the edge mutation change
                    mutation_attempts++; 
                } else {
                    mutation_node->target_node = new_target; 
                    mutate_success = 1;
                    if (record_changes){
                        cur_mutation->mutate_node_tgt = new_target;
                        cur_mutation->new_weight = mutation_node->weight;
                        cur_mutation->mutation_type= EDGE;
                    }
                }
            }

            // Switch to weight mutation
            if (!mutate_success){
                mutation=WEIGHT; 
            }
        }   
        // Switch weight..
        if (mutation == WEIGHT){
            mutation_node->weight = rand_double();
            if (record_changes){
                cur_mutation->mutate_node_tgt = mutation_node->target_node;
                cur_mutation->new_weight = mutation_node->weight;
                cur_mutation->mutation_type = WEIGHT;
            }
        }
    } 
    return new_net; 
}


/* 
 * Free a given linked list 
 */
int free_chain(node* my_node){
    if (!my_node->next){
        free(my_node); 
        return 1;
    } else { 
        int freed =  free_chain(my_node->next); 
        free(my_node); 
        return freed + 1; 
    }
}

/* 
 * Free a given linked list of mutations 
 */
int free_mutation_list(mutation_list* my_node){
    if (!my_node->next){
        free(my_node); 
        return 1;
    } else { 
        int freed =  free_mutation_list(my_node->next); 
        free(my_node); 
        return freed + 1; 
    }
}


/* 
 * Copy mutation list into the spot in memory given 
 */
void copy_mutation_list(mutation_list* list_to_copy, mutation_list** mem){
    mutation_list* cur_node = list_to_copy;
    // Current node in the list we create
    mutation_list* cur_node_new_list = NULL;
    // Start of the new list.
    mutation_list* head_new_list = NULL; 
    while (cur_node){
        if (head_new_list){
            cur_node_new_list->next = malloc(sizeof(mutation_list));
            cur_node_new_list = cur_node_new_list->next; 
        } else {
            head_new_list = malloc(sizeof(mutation_list));
            cur_node_new_list = head_new_list; 
        }
        cur_node_new_list->mutate_node_src = cur_node->mutate_node_src;
        cur_node_new_list->mutate_node_tgt = cur_node->mutate_node_tgt;
        cur_node_new_list->mutation_type = cur_node->mutation_type;
        cur_node_new_list->mutate_node_src = cur_node->mutate_node_src;
        cur_node_new_list->new_weight = cur_node->new_weight;
        cur_node_new_list->next = NULL; 
        cur_node = cur_node->next; 
    }
    *mem = head_new_list;     
}


/* 
 * Return the number of nodes in the network... 
 * Free them all
 */
int free_network(network* net, int n){
    int ctr = 0; 
    // int test; 
    for (int i = 0; i < n; i ++){
        node* current_node = net->nodes[i]; 
        if (current_node){
            // test = free_chain(current_node);
            // printf("%d, ", test); 
            ctr += free_chain(current_node); 
        }
    }
    // printf("\n\n"); 
    free(net->nodes); 
    free(net->num_outputs);
    if (net->mutation_list){
        free_mutation_list(net->mutation_list);
    }
    free(net); 
    return ctr;
}

/* 
 * Random uniform in range -1, 1
 */
double rand_double(void){
    
    // float in range -1 to 1
    return (double)rand()/RAND_MAX*2.0-1.0;
}


/* 
 * Random uniform in range 0, 1
 */
double rand_num(void){
    
    // float in range 0 to 1
    return (double)rand()/RAND_MAX; 
}


/* 
 * Return random boolean
 */ 
int rand_bool(void){
 
    return rand() % 2;
}

/* 
 * Return random in a range [low, high)
 */
int rand_int_range(int low, int high){
    
    return rand() % (high - low) + low;
}

/* 
 * Return random in a range [low, high)
 */
int* rand_network_state(int n){
    int* state = malloc(sizeof(int) * n); 

    for (int i = 0; i < n; i++){
        state[i] = rand_bool(); 
    }
    return state; 
}

/* 
 * Generate a random target function
 */ 
int* rand_target(int length) {
    int* target = malloc(sizeof(int) * length); 

    for (int j = 0; j < length; j++){
        target[j] = rand_bool(); 
    }

    return target; 
}

/* 
 * Calculate the LCM of two numbers
 * Taken from online... 
 */
int lcm(int n1, int n2){
    int min = n1 > n2 ? n2 : n1; 
    int r; 
    for(int i = min; i > 0; i--){
        // Checks if i is factor of both integers
        if(n1%i==0 && n2%i==0){
            r = (n1*n2)/i;
            return r; 
        }
    }
    return n1 * n2; 
}

/* 
 * Move the top "num_winners" scoring networks into the first 50 positions 
 * Uses truncated selection sort
 * population_capacity is how many items in scores an poulation
 */
void sort_winners(int num_winners, int population_capacity,
                  network** population, double* scores){
    int max_index;
    network* temp_network;  
    double  temp_double; 
    for (int j = 0; j < num_winners; j++)
    {
        /* find the min element in the unsorted a[j .. n-1] */

        /* assume the min is the first element */
        max_index = j;
        /* test against elements after j to find the smallest */
        for (int i = j+1; i < population_capacity; i++)
        {
            /* if this element is greater, then it is the new maximum */
            if (scores[i] > scores[max_index])
            {
                /* found new minimum; remember its index */
                max_index = i;
                // Randomly switch them if the same... 
            }
            // } else if (scores[i] == scores[max_index] && rand_bool()){
            //     max_index = i;
            // }
        }

        // Ensure that we switch for a real network!
        if (max_index != j && population[max_index]) 
        {
            // Swap networks j and min_index
            temp_network = population[j];
            population[j] = population[max_index];
            population[max_index] = temp_network; 

            temp_double = scores[j]; 
            scores[j] = scores[max_index];
            scores[max_index] = temp_double; 
        }
    }
}

/* 
 * Linear search
 * Return 1 if any list overlapin list, else 0
 */
int linear_search(int* list, int list_length, int item){
    for (int i = 0; i < list_length; i++){
        if (list[i] == item){
            return 1; 
        }
    }
    return 0; 
}


/* 
 * Calculate average
 */
double calculate_average(double* values, int length){
    double running_sum = 0; 
    for (int i = 0; i < length; ++i){
        running_sum += values[i]; 
    }
    return running_sum / length; 
}

/* 
 * Get argmax of list
 */
int arg_max(int* list, int list_length){
    int max = list[0]; 
    int max_index = 0; 
    for (int i = 0; i < list_length; i++){
        if (list[i] > max){
            max_index = i; 
            max = list[i]; 
        }
    }
    return max_index; 
}

/* 
 * Check if a directory exists, if it doesn't make it
 */
void check_make_dir(char* my_dir){
    DIR* dir = opendir(my_dir);
    if (dir){
        /* Directory exists. */
        closedir(dir);
    } else {
        mkdir(my_dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
}


/* 
 * Get a date time stamp...
 */
char* get_string_time(void){
    char* time_buffer = malloc(sizeof(char) * 255);
    time_t rawtime = time(NULL);
    struct tm* timeinfo;
    timeinfo = localtime (&rawtime);
    //time (&rawtime);

    strftime (time_buffer, 64, "%F_%X", timeinfo);
    return time_buffer; 
}

/* 
 *  Write out the mutation list to file for this network
 */
void write_out_mutations(char* file_name, int generations, mutation_list** mut_list){
    char* directory = "mutation_recordings/";
    check_make_dir(directory);  
    char my_file[255];
    char* string_time = get_string_time(); 
    sprintf(my_file,"%s%s_%s%s", directory, file_name, string_time, ".json");
    FILE* out_file = fopen(my_file, "w");

    mutation_list* cur_node; 
    fprintf(out_file, "{\n");

    for (int g = 0; g < generations; g++){
        // If > 0, print the comma from the preceding item. 
        if (g > 0){
            fprintf(out_file, ",");
        }
        fprintf(out_file, "\"%d\": ", g);
        fprintf(out_file, "[");
        cur_node = mut_list[g];
        while (cur_node){
            fprintf(out_file, "{");
            fprintf(out_file, "\"source\" : %d\n,", cur_node->mutate_node_src);
            fprintf(out_file, "\"target\" : %d\n,", cur_node->mutate_node_tgt);
            fprintf(out_file, "\"mutation_type\" : %d,\n", cur_node->mutation_type);

            fprintf(out_file, "\"new_weight\" : %.4f\n", cur_node->new_weight);

            fprintf(out_file, "}\n"); 
            if (cur_node->next){
                fprintf(out_file, ", "); 
            }
            cur_node = cur_node->next; 

        }

        if (mut_list[g]){
            free_mutation_list(mut_list[g]);
        }

        fprintf(out_file, "]");
        cur_node = NULL; 
    }
    fprintf(out_file, "}");     
    fclose(out_file);
}


/* 
 * Helper function to wirte all our output to json out file
 */
void write_out_files(char* file_name, network* net, int** target_fns, int* target_lengths,
                     int num_targets, int target_node, int generations,
                     int population_size, int selection_delay, int sort_first,
                     hub_behavior behavior, double avg_score, int MSE, int* periods,
                     int** stereotypes){


    char* directory = "output_networks/";
    check_make_dir(directory);  
    char my_file[255];
    char* string_time = get_string_time(); 
    sprintf(my_file,"%s%s_%s%s", directory, file_name, string_time, ".json");


    FILE* out_file = fopen(my_file, "w");
    fprintf(out_file, "{\n");
    fprintf(out_file, "\"date\": \"%s\",\n", string_time);
    fprintf(out_file, "\"num_targets\": %d,\n", num_targets);
    fprintf(out_file, "\"hub_node\": %d,\n", net->oscil_node);
    fprintf(out_file, "\"target_node\": %d,\n", target_node);
    fprintf(out_file, "\"n\": %d,\n", net->num_nodes);
    fprintf(out_file, "\"generations\": %d,\n", generations);
    fprintf(out_file, "\"population_size\": %d,\n", population_size);
    fprintf(out_file, "\"selection_delay\": %d,\n", selection_delay);
    fprintf(out_file, "\"sort_first\": %d,\n", sort_first);
    fprintf(out_file, "\"hub_behavior\": %d,\n", behavior);
    fprintf(out_file, "\"score_at_end\": %.2f,\n", avg_score);

    // Put target lengths in... 
    fprintf(out_file, "\"target_lengths\": [%d", target_lengths[0]);
    for (int i = 1; i < num_targets; i++){
        fprintf(out_file, ", "); 
        fprintf(out_file, "%d", target_lengths[i]);
    }
    fprintf(out_file, "],\n"); 

    // Put oscil periods in... 
    if (periods){
        fprintf(out_file, "\"oscil_periods\": [%d", periods[0]);
        for (int i = 1; i < num_targets; i++){
            fprintf(out_file, ", "); 
            fprintf(out_file, "%d", periods[i]);
        }
        fprintf(out_file, "],\n");
    }

    // Put stereotypes in 
    if (stereotypes[0]){
        fprintf(out_file, "\"stereotype_functions\": ["); 
        // Now print out all the target functions
        for (int i = 0; i < num_targets; i++){
            // loop over length of this target
            fprintf(out_file, "["); 
            for (int j = 0; j < target_lengths[i]; j++){
                // print each output followed by comma, until last one -- 
                // Follow that with square bracket 
                fprintf(out_file, "%d", stereotypes[i][j]);
                if (j != (target_lengths[i] - 1)) {
                    fprintf(out_file, ", "); 
                } else {
                    fprintf(out_file, "]"); 
                }
            }
            // if we have finished writing all the targets, close with square bracket
            // Else: add comma and write next one 
            if (i == num_targets - 1 ){
                fprintf(out_file, "],\n");

            } else { 
                fprintf(out_file, ",\n");
            }
        }
    }
    
    fprintf(out_file, "\"target_functions\": ["); 
    // Now print out all the target functions
    for (int i = 0; i < num_targets; i++){
        // loop over length of this target
        fprintf(out_file, "["); 
        for (int j = 0; j < target_lengths[i]; j++){
            // print each output followed by comma, until last one -- 
            // Follow that with square bracket 
            fprintf(out_file, "%d", target_fns[i][j]);
            if (j != (target_lengths[i] - 1)) {
                fprintf(out_file, ", "); 
            } else {
                fprintf(out_file, "]"); 
            }
        }
        // if we have finished writing all the targets, close with square bracket
        // Else: add comma and write next one 
        if (i == num_targets - 1 ){
            fprintf(out_file, "],\n");

        } else { 
            fprintf(out_file, ",\n");
        }
    }




    // Write matrix!!

    fprintf(out_file, "\"network\": ["); 
    int n = net->num_nodes;
    double* my_mat = malloc(n * n * sizeof(double)); 
    // Move the network to matrix form
    net_to_mat(net, my_mat); 


    // Now print out all the target functions
    for (int i = 0; i < n; i++){
        // loop over length of this target
        fprintf(out_file, "["); 
        for (int j = 0; j < n; j++){
            // print each output followed by comma, until last one -- 
            // Follow that with square bracket 
            fprintf(out_file, "%.4f", my_mat[i*n + j]);
            if (j != (n - 1)) {
                fprintf(out_file, ", "); 
            } else {
                fprintf(out_file, "]"); 
            }
        }
        // if we have finished writing all the targets, close with square bracket
        // Else: add comma and write next one 
        if (i == n - 1 ){
            fprintf(out_file, "],\n");

        } else { 
            fprintf(out_file, ",\n");
        }
    }
    ///////////////////////////////////////////
    // Easy hack to add one last term at the end to not handle the comma in the loop 
    fprintf(out_file, "\"MSE\": %d\n", MSE);
    fprintf(out_file, "}\n"); 
    fclose(out_file); 
    free(string_time); 
    free(my_mat); 


}





