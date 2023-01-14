#ifndef HELPERS_H_INCLUDED
#define HELPERS_H_INCLUDED

#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <time.h> 
#include <errno.h>
#include <sys/stat.h>

#define MUTATION_THRESHOLD 0.02


typedef enum mutation_type {NONE, WEIGHT, EDGE} mutation_type; 
typedef enum hub_behavior{FREE, OSCILLATE, BLOCK_ON, BLOCK_OFF, STEREOTYPED, RANDOM, NO_FREQ} hub_behavior; 

/* 
 * Structure to hold a list of mutations that the network accumulated on the last step
 */ 
typedef struct mutation_list{
    int mutate_node_src; 
    int mutate_node_tgt;
    mutation_type mutation_type;
    double new_weight;
    struct mutation_list* next;
} mutation_list;


/* 
 * Create adjacency list structs
 */
typedef struct node { 
    double weight;
    int target_node; 
    struct node* next; 
} node; 

/* 
 * Define a separate table to alias all output nodse
 * Use this in order to more easily preform swap mutations
 */

/* 
 * Create adjacency list structs
 * NOTE: THis is an adjacency list of the OUTGOING connections
 */
typedef struct network { 
    int num_nodes;  
    int oscil_node;
    int* num_outputs;
    node** nodes;
    mutation_list* mutation_list; 
} network; 

/* 
 * Deep copy a network.. 
 */ 
network* copy_network(network* net); 


/* 
 * Copy mutation list into the spot in memory given 
 */
void copy_mutation_list(mutation_list* list_to_copy, mutation_list** mem);

/* 
 * Convert an adjacency list to a matrix representation
 */ 
void net_to_mat(network* net, double* mat); 

/* 
 * Mutate the network
 * TODO: Add support for actually switching around edges..
 * 
 */
network* mutate_network(network* old_net, int record_changes);

/* 
 * Convert matrix into adjacency list... 
 */ 
network* convert_mat(double* array, int n); 

int free_chain(node* my_node);

/* 
 * Free mutation list
 */
int free_mutation_list(mutation_list* my_node); 

/* 
 * Return the number of nodes in the network... 
 * Free them all
 */
int free_network(network* net, int n); 

/* 
 * Random uniform in range -1, 1
 */
double rand_double(void);

/* 
 * Return random boolean
 */ 
int rand_bool(void); 

/* 
 * Return random network state... 
 */ 
int* rand_network_state(int n); 

/* 
 * Copy a random state of a network
 */ 
int* copy_network_state(int n, int* net_state); 

/* 
 * Perturb random state
 */
void perturb_start(int n, int* net_state, double p); 

// Rand between 0 and 1
double rand_num(void);

// Rand int, range [low, high)
int rand_int_range(int low, int high);

/* 
 * Random target 
 */ 
int* rand_target(int length); 

/* 
 * Calc LCM  
 */ 
int lcm(int n1, int n2); 

/* 
 * Move the top "num_winners" scoring networks into the first 50 positions 
 * Uses truncated selection sort
 */
void sort_winners(int num_winners, int population_capacity, network** population, double* scores); 

/* 
 * Linear search
 * Return 1 if in list, else 0
 */
int linear_search(int* list, int list_length, int item);

/* 
 * Calculate average
 */
double calculate_average(double* values, int length); 

/* 
 * Get argmax of list
 */
int arg_max(int* list, int list_length); 


/* 
 * Check if a directory exists, if it doesn't make it
 */
void check_make_dir(char* my_dir);

/* 
 * Get a date time stamp...
 */
char* get_string_time(void);

/* 
 *  Write out the mutation list to file for this network
 */
void write_out_mutations(char* file_name, int generations, mutation_list** mut_list);


/* 
 * Helper function to wirte all our output to json out file
 */
void write_out_files(char* file_name, network* net, int** target_fns, int* target_lengths,
                     int num_targets, int target_node, int generations,
                     int population_size, int selection_delay, int sort_first,
                     hub_behavior behavior, double avg_score, int MSE, int* periods,
                     int** stereotypes); 

#endif
