/*
test.c

Write tests and example usages of scoring.c

*/


#include "scoring.h"


/* 
 * Generate adjacency matrices 
 */
double* generate_adjacency_matrices(int n, int num_networks){
    double* my_mat = malloc(sizeof(double) * n * n * num_networks);
    int ctr; 
    for (int net = 0; net < num_networks; net++){
        ctr = 0; 
        for (int j = 0; j < n; j++){
            for (int i = 0; i < n; i++){
                if (rand_num() > 0.940){
                    my_mat[n*n*net + j*n + i] = rand_double();
                    ctr++; 
                } else{
                    my_mat[n*n*net + j*n + i] = 0; 
                }
            }
        }
        // printf("Number edges: %d\n", ctr); 
    }

    return my_mat; 
}


/* 
 * Test simple update
 */

void test_update(){
    int n = 1000;
    int time_steps = 100;
    int* bools_ar = rand_network_state(n);
    double* my_mat = generate_adjacency_matrices(n, 1); 

    // Generate return table 
    int* return_table = malloc(sizeof(int) * n * time_steps);
    for (int i = 0 ; i < n * time_steps; i++){
        return_table[i] = 0; 
    }

    int node_index = 0; 
    int period = 1; 
    hub_behavior behavior = FREE; 

    // // Matrix mult 
    // // Adj list conversion
    network* net= convert_mat(&my_mat[0], n); 
    
    clock_t begin, end;
    begin = clock();
    c_update(net, &bools_ar[0], &return_table[0], n, time_steps, 
             node_index, period, behavior, 1); 
    end = clock();

    printf("FREED:%d\n", free_network(net, n)); 
    free(return_table); 

    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("C Clock time: %f\n", time_spent);
}

/* 
 * Test simple network evolution
 */
void test_evolution(){
    int n = 100;
    int time_steps = 1000;
    int num_replicates = 3; 
    int pop_size = 50;
    int generations=100; 
    double scores[generations]; 
    int target_length = 10; 
    int sort_first = 1; 

    double* my_mat = generate_adjacency_matrices(n, pop_size); 

    for (int i =0; i < generations; i++)
        scores[i] = 0; 

    c_evolution_wrapper_free(my_mat, n, time_steps, generations, 
                             scores, pop_size, num_replicates,
                             target_length, -1, sort_first, 0, NULL, 0, 1); 
}

/* 
 * Test simple network evolution
 */
void test_evolution_oscil(){
    int n = 100;
    int time_steps = 1000;
    int num_replicates = 3; 
    int pop_size = 50;
    int generations=100; 
    int period = 10; 
    int node_indices[pop_size];  
    double scores[generations]; 
    int target_length = 10; 
    int sort_first = 1;

    double* my_mat = generate_adjacency_matrices(n, pop_size); 

    // Randomly set them all to the same thing.. 
    for (int i = 0; i < pop_size; i++)
        node_indices[i] = 10; 
    for (int i =0; i < generations; i++){
        scores[i] = 0; 
    }
    c_evolution_wrapper_oscil(my_mat, n, time_steps, generations,
                              node_indices, period, scores, pop_size,
                              num_replicates, target_length, -1, sort_first, 0, 0,
                              NULL, 0, 1); 

}

/* 
 * Test simple network evolution
 */
void test_evolution_oscil_multiplex(){
    int n = 100;
    int time_steps = 1000;
    int num_replicates = 3; 
    int pop_size = 50;
    int generations=3000; 
    int num_functions = 2;
    int periods[] = {10,12,14}; 
    int node_indices[pop_size];  
    double scores[generations]; 
    int target_lengths[] = {10,10,10}; 
    int sort_first = 1; 
    int selection_delay = sizeof(target_lengths) / sizeof(target_lengths[0]); 
    int MSE = 1; 
    int learning_blocks = 1; 
    int include_input = 0;
    int check_mini_cycles = 0; 
    double* my_mat = generate_adjacency_matrices(n, pop_size); 


    // Randomly set them all to the same thing.. 
    for (int i = 0; i < pop_size; i++)
        node_indices[i] = 10; 
    for (int i =0; i < generations; i++){
        scores[i] = 0; 
    }
    c_evolution_multiplex_wrapper_oscil(my_mat, n, time_steps, generations,
                                        node_indices, periods, scores, pop_size, num_replicates,
                                        target_lengths, -1, num_functions, selection_delay, sort_first,
                                        MSE, learning_blocks, NULL, 0, NULL, 0, NULL, check_mini_cycles, include_input, -1); 
    for (int i = 0; i < generations; i += 1){
        printf("Score for generation %d: %0.2f\n", i, scores[i]); 
    }
}

/* 
 * Test random scores... 
 */
int test_random_scores(){
    int n = 10000; 
    int* target; 
    int* cycle;
    int target_length = 2; 
    int cycle_length = 2; 
    double score = 0; 
    double temp_score = 0; 
    for (int q = 1; q < 2; q += 5){
        target_length = 2; 
        cycle_length = 2;  
	//	for (int m = 0; m < target_length; m++){
	//  printf("%d\t", target[m]);
	//}
	    printf("\n");
        for (int i = 0; i < n; i ++){    
            cycle = rand_target(cycle_length); 
	    target = rand_target(target_length);	    
	    temp_score = score_cycle(target, cycle, target_length, cycle_length); 
	    printf("TEMPORARY SCORE: %.2f\n", temp_score); 
            score += temp_score; 
            free(target); 
            free(cycle); 
        } 
        printf("Average fitness for random targ of size %d and random cycle of size %d: %.2f\n", 
	       target_length, cycle_length, score / n); 
        score = 0; 
    }
    return 0; 
}

/* 
 * Example of running this directly without cython.. 
 */
int main(){
    int seed = time(NULL);
    srand(seed); 
    test_random_scores();
    // test_update();
    // test_evolution();  
    // test_evolution_oscil();
    // test_evolution_oscil_multiplex();   

}
