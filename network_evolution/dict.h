#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED
/* ^^ these are the include guards */

/* Prototypes for the functions */

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

// Hash function... 
#define FNV_PRIME_32 16777619
#define FNV_OFFSET_32 2166136261U
#define HASH_SIZE 1024

typedef struct entry_t {
    bool *key;
    int value;
    struct entry_t *next;
} entry_t;


typedef struct hashtable_t {
    int size; 
    int node_length; 
    struct entry_t **table; 

} hashtable_t; 

uint32_t FNV32(bool* s, int size); 

void hash_testing(int* ar, int n); 

int ht_check_insert( hashtable_t *hashtable, entry_t* entry, uint32_t bin); 

hashtable_t *ht_create(int size, int node_length); 

int free_entry_chain(entry_t* entry);
int free_hash_table(hashtable_t* hashtable); 

#endif