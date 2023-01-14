"""
scoring_function.pyx

simple cython 
"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
from cpython.ref cimport PyObject
from libc.stdint cimport uintptr_t


# declare the interface to the C code
cdef extern void c_update_wrapper (double* array, int* state, 
        int* state_table, int n, int time_steps, int* r_seed, int include_input)

cdef extern void c_update_oscillation_wrapper (double* array, int* state, 
        int* state_table, int n, int time_steps, int node_index, int period,
        int* r_seed, int include_input)

########################################################################

cdef extern void c_evolution_wrapper_free(double* array, int n, int time_steps, int generations, 
                              double* scores, int pop_size, int num_replicates, int target_length, int target_node,
                              int sort_first, int record_mutations, int* r_seed, int check_mini_cycles,
                              int include_input)


cdef extern void c_evolution_wrapper_random(double* array, int n, int time_steps, int generations,
                                int* node_indices, double* scores, int pop_size, int num_replicates,
                                int target_length, int target_node, int sort_first, int record_mutations, int* r_seed,
                                int check_mini_cycles, int include_input)



cdef extern void c_evolution_wrapper_no_freq(double* array, int n, int time_steps, int generations,
                               int* node_indices, double* scores, int pop_size,
                               int num_replicates, int target_length, int target_node, int sort_first,
                               int record_mutations, int* r_seed, int check_mini_cycles, 
                               int include_input)

cdef extern void c_evolution_wrapper_oscil(double* array, int n, int time_steps, int generations,
                               int* node_indices, int period, double* scores, int pop_size,
                               int num_replicates, int target_length, int target_node, int sort_first,
                               int stereotyped, int record_mutations, int* r_seed, 
                               int check_mini_cycles, int include_input)


########################################################################

cdef extern void c_evolution_multiplex_wrapper_free(double* array, int n, int time_steps, int generations,
                                        double* scores, int pop_size, int num_replicates,
                                        int* target_lengths, int target_node, int num_functions, int selection_delay,
                                        int sort_first, int MSE, int learning_blocks,
                                        int* learning_rotation, double* record_all, int record_mutations,
                                        int* r_seed, int check_mini_cycles, int include_input)


cdef extern void c_evolution_multiplex_wrapper_no_freq(double* array, int n, int time_steps, int generations,
                                         int* node_indices, double* scores, int pop_size, int num_replicates,
                                         int* target_lengths, int target_node, int num_functions, int selection_delay,
                                         int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, double* record_all, int record_mutations,
                                         int* r_seed, int check_mini_cycles, int include_input)

cdef extern void c_evolution_multiplex_wrapper_oscil(double* array, int n, int time_steps, int generations,
                                         int* node_indices, int* periods, double* scores, int pop_size,
                                         int num_replicates, int* target_lengths, int target_node, int num_functions,
                                         int selection_delay, int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, int stereotyped, double* record_all,
                                         int record_mutations, int* r_seed, int check_mini_cycles, int include_input, int switch_time) 

cdef extern void c_evolution_multiplex_fixed_starts(double* array, int n, int time_steps, int generations,
                                         int* node_indices, double* scores, int pop_size, int num_replicates,
                                         int* target_lengths, int target_node, int num_functions, int selection_delay,
                                         int sort_first, int MSE, int learning_blocks,
                                         int* learning_rotation, double* record_all, int record_mutations,
                                         int* r_seed, int include_input, double p, int constant_hub)

# Remove some checks that cython uses
# Assumes square matrix input
# Use this to check adjacency list implementation time
@cython.boundscheck(False)
@cython.wraparound(False)
def update_wrapper(np.ndarray[double, ndim=2, mode="c"] input not None, 
                np.ndarray[int, ndim=1, mode="c"] state not None, 
                np.ndarray[int, ndim=2, mode="c"] return_table not None, int r_seed, int include_input):

        cdef int n = input.shape[0]
        cdef int time = return_table.shape[0] 
        if (return_table.shape[1] != n):
                raise Exception("Invalid return table shape. Time table must have same dimensions as network")
        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL

        c_update_wrapper (&input[0,0], &state[0], &return_table[0,0], n, time, seed, include_input)

        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def update_oscillation_wrapper(np.ndarray[double, ndim=2, mode="c"] input not None, 
                np.ndarray[int, ndim=1, mode="c"] state not None, 
                np.ndarray[int, ndim=2, mode="c"] return_table not None, int node_index, int period,
                int r_seed, int include_input):

        cdef int n = input.shape[0]
        cdef int time = return_table.shape[0] 
        if (return_table.shape[1] != n):
                raise Exception("Invalid return table shape. Time table must have same dimensions as network")

        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL


        c_update_oscillation_wrapper (&input[0,0], &state[0], &return_table[0,0], 
                n, time, node_index, period, seed, include_input)

        return None

# @cython.boundscheck(False)
# @cython.wraparound(False)
def evolve_free(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                int target_length, int target_node, int sort_first, int num_replicates, int record_mutations,
                int r_seed, int check_mini_cycles, int include_input):

        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]

        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL


        c_evolution_wrapper_free (&input[0,0,0], n, time_steps, 
                generations, &scores[0], pop_size, num_replicates, target_length, target_node,
                sort_first, record_mutations, seed, check_mini_cycles, include_input) 

        return None


@cython.boundscheck(False)
@cython.wraparound(False)
def evolve_random(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                np.ndarray[int, ndim=1, mode="c"] node_indices not None,
                int target_length, int target_node, int sort_first, int num_replicates, 
                int record_mutations, int r_seed, int check_mini_cycles, int include_input):
        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]

        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL

        c_evolution_wrapper_random (&input[0,0,0], n, time_steps, generations, 
                &node_indices[0], &scores[0], pop_size, num_replicates, target_length, target_node, 
                sort_first, record_mutations, seed, check_mini_cycles, include_input) 

        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def evolve_no_freq(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                np.ndarray[int, ndim=1, mode="c"] node_indices not None, int period,
                int target_length, int target_node, int sort_first, int num_replicates, 
                int record_mutations, int r_seed, int check_mini_cycles, int include_input):
        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]

        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL
        
        c_evolution_wrapper_no_freq (&input[0,0,0], n, time_steps, generations, 
                &node_indices[0], &scores[0], pop_size, num_replicates, 
                target_length, target_node, sort_first, record_mutations, seed, check_mini_cycles, include_input)

        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def evolve_oscillation(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                np.ndarray[int, ndim=1, mode="c"] node_indices not None, int period,
                int target_length, int target_node, int sort_first, int stereotyped, int num_replicates, 
                int record_mutations, int r_seed, int check_mini_cycles, int include_input):
        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]

        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL
        

        c_evolution_wrapper_oscil (&input[0,0,0], n, time_steps, generations, 
                &node_indices[0], period, &scores[0], pop_size, num_replicates, 
                target_length, target_node, sort_first, stereotyped, record_mutations, seed,
                check_mini_cycles, include_input)

        return None


@cython.boundscheck(False)
@cython.wraparound(False)
def multiplex_free(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                np.ndarray[int, ndim=1, mode="c"] target_lengths, int target_node, int selection_delay, int sort_first, int MSE, 
                int learning_blocks, np.ndarray[int, ndim=1, mode="c"] learning_rotation,
                np.ndarray[double, ndim=2, mode="c"] record_all, int num_replicates, int record_mutations, 
                int r_seed, int check_mini_cycles, int include_input):



        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]
        cdef int num_functions = target_lengths.shape[0]
        cdef int mse_val = 1 if MSE else 0
        cdef int* c_learning_rotation; 
        if learning_rotation is not None: 
                c_learning_rotation = NULL
        else: 
                c_learning_rotation = &learning_rotation[0]

        cdef double* recording; 
        if record_all is not None:
            recording = &record_all[0,0]
        else:
            recording = NULL

        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL


        c_evolution_multiplex_wrapper_free(&input[0,0,0], n, time_steps, 
                generations, &scores[0], pop_size, num_replicates, &target_lengths[0], target_node, num_functions,
                selection_delay, sort_first,  mse_val, learning_blocks, c_learning_rotation,
                recording, record_mutations, seed, check_mini_cycles, include_input)

        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def multiplex_no_freq(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                np.ndarray[int, ndim=1, mode="c"] node_indices not None, 
                np.ndarray[int, ndim=1, mode="c"] target_lengths, int target_node, int selection_delay, 
                int sort_first, int MSE, int learning_blocks, np.ndarray[int, ndim=1, mode="c"] learning_rotation,
                np.ndarray[double, ndim=2, mode="c"] record_all, int num_replicates, int record_mutations, int r_seed, 
                int check_mini_cycles, int include_input):

        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]
        cdef int num_functions = target_lengths.shape[0]
        cdef int mse_val = 1 if MSE else 0
        cdef int* c_learning_rotation; 
        if learning_rotation is not None: 
            c_learning_rotation = NULL
        else: 
            c_learning_rotation = &learning_rotation[0]

        cdef double* recording; 
        if record_all is not None:
            recording = &record_all[0,0]
        else:
            recording = NULL
            
        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL

        c_evolution_multiplex_wrapper_no_freq(&input[0,0,0], n, time_steps, 
                generations, &node_indices[0], &scores[0], 
                pop_size, num_replicates, &target_lengths[0], target_node, num_functions, selection_delay,
                sort_first, mse_val, learning_blocks, c_learning_rotation,
                recording, record_mutations, seed, check_mini_cycles, include_input)

        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def multiplex_oscil(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                np.ndarray[int, ndim=1, mode="c"] node_indices not None, 
                np.ndarray[int, ndim=1, mode="c"] periods,
                np.ndarray[int, ndim=1, mode="c"] target_lengths, int target_node, int selection_delay, 
                int sort_first, int MSE, int learning_blocks, np.ndarray[int, ndim=1, mode="c"] learning_rotation, int stereotyped,
                np.ndarray[double, ndim=2, mode="c"] record_all, int num_replicates, int record_mutations, int r_seed, 
                int check_mini_cycles, int include_input, int switch_time):

        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]
        cdef int num_functions = target_lengths.shape[0]
        cdef int mse_val = 1 if MSE else 0
        cdef int* c_learning_rotation; 
        if learning_rotation is not None: 
            c_learning_rotation = NULL
        else: 
            c_learning_rotation = &learning_rotation[0]

        cdef double* recording; 
        if record_all is not None:
            recording = &record_all[0,0]
        else:
            recording = NULL
            
        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL

        c_evolution_multiplex_wrapper_oscil(&input[0,0,0], n, time_steps, 
                generations, &node_indices[0], &periods[0], &scores[0], 
                pop_size, num_replicates, &target_lengths[0], target_node, num_functions, selection_delay,
                sort_first, mse_val, learning_blocks, c_learning_rotation, stereotyped,
                recording, record_mutations, seed, check_mini_cycles, include_input, switch_time)

        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def multiplex_fixed_start(np.ndarray[double, ndim=3, mode="c"] input not None, 
                np.ndarray[double, ndim=1, mode="c"] scores not None, int time_steps, 
                np.ndarray[int, ndim=1, mode="c"] node_indices not None, 
                np.ndarray[int, ndim=1, mode="c"] target_lengths, int target_node, int selection_delay, 
                int sort_first, int MSE, int learning_blocks, np.ndarray[int, ndim=1, mode="c"] learning_rotation,
                np.ndarray[double, ndim=2, mode="c"] record_all, int num_replicates, int record_mutations, int r_seed, 
                int include_input, double p, int constant_hub):

        cdef int n = input.shape[1]
        cdef int generations = scores.shape[0] 
        cdef int pop_size = input.shape[0]
        cdef int num_functions = target_lengths.shape[0]
        cdef int mse_val = 1 if MSE else 0
        cdef int* c_learning_rotation; 
        if learning_rotation is not None: 
            c_learning_rotation = NULL
        else: 
            c_learning_rotation = &learning_rotation[0]

        cdef double* recording; 
        if record_all is not None:
            recording = &record_all[0,0]
        else:
            recording = NULL
            
        cdef int* seed
        if r_seed != 0:
            seed = &r_seed
        else:
            seed = NULL

        c_evolution_multiplex_fixed_starts(&input[0,0,0], n, time_steps, generations,
                &node_indices[0], &scores[0], pop_size, num_replicates,
                &target_lengths[0], target_node, num_functions, selection_delay,
                sort_first, mse_val, learning_blocks, c_learning_rotation, recording, record_mutations,
                seed, include_input, p, constant_hub)
        
        return None


