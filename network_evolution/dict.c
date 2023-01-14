/* In general it's good to include also the header of the current .c,
   to avoid repeating the prototypes */
#include "dict.h"



uint32_t FNV32(bool* s, int size){
    uint32_t hash = FNV_OFFSET_32; 
    for (int i = 0; i < size; i++)
    {
        hash = hash ^ (s[i]); // xor next byte into the bottom of the hash
        hash = hash * FNV_PRIME_32; // Multiply by prime number found to work well
    }
    // Hardcode hash size... 
    return hash & (HASH_SIZE- 1);
} 


/* 
 * Example of converting bool array to string..
 */
 void hash_testing(int* ar, int n){
    bool bools_ar[n];
    // Set state_array[0:n] = state, the initial conditions
    for (int j = 0; j < n; j++){
        bools_ar[j] = (bool) ar[j]; 
    }

    printf("Bool hash value: %u\n", FNV32(bools_ar, n)); 

    printf("Size of the bools ar: %lu\n", sizeof(bools_ar) / sizeof(bools_ar[0]));
    printf("Size of the bool type: %lu\n", sizeof(bool));


 }

hashtable_t *ht_create(int size, int node_length) {
    hashtable_t *hashtable = NULL; 

    if (size < 1) return NULL; 
    // allocate table
    if( ( hashtable = malloc( sizeof( hashtable_t ) ) ) == NULL ) {
        return NULL;
    }

    /* Allocate pointers to the head nodes. */
    if( ( hashtable->table = malloc( sizeof( entry_t * ) * size ) ) == NULL ) {
        return NULL;
    }
    for(int i = 0; i < size; i++ ) {
        hashtable->table[i] = NULL;
    }

    hashtable->size = size;
    hashtable->node_length = node_length; 
    return hashtable;   
}

/*
 *  Insert a key-value pair (entry_t) into the hash table. 
 *  Return the value if we seen this already; return -1 if we haven't
 *  Provide hash bin (since we can calculate dynamically), entry, and hashtable..
 */
int ht_check_insert( hashtable_t *hashtable, entry_t* entry, uint32_t bin) {

    // Get current bucket 
    entry_t* next = hashtable->table[bin];
    entry_t* last = NULL; 

    // Get current key and num nodes for memcmp
    bool* key = entry->key; 
    int num_nodes = hashtable->node_length;  

    // If we have a non null start, a place to go next, and the next one is not equal
    // Corner case: What if the first entry is equal?? 
    while( next != NULL && next->key != NULL && memcmp(key, next->key, num_nodes) != 0 ) {
        last = next;
        next = next->next;
    }

    /* Found a duplicate  */
    if( next != NULL && next->key != NULL && memcmp( key, next->key, num_nodes) == 0 ) {
        // return next-> value.. 
        return next->value; 
    /* Nope, could't find it.  Time to grow a pair. */
    } else {

        // We're at the start of the linked list in this bin. 
        if( next == hashtable->table[ bin ] ) {
            entry->next = next;
            hashtable->table[ bin ] = entry;

        /* We're at the end of the linked list in this bin. */
        } else if ( next == NULL ) {
            last->next = entry;
        } else {
            // We will never be in the middle of the linkedlist... 
            printf("Error in hash table\n"); 
            exit(1); 
        }
        return -1; 
    }
}

/* 
 * free_chain Returns number freed
 */
int free_entry_chain(entry_t* entry){
    if (!entry->next){
    	free(entry->key);
        free(entry); 
        return 1;
    } else { 
        int freed =  free_entry_chain(entry->next); 
        free(entry->key); 
        free(entry); 
        return freed + 1; 
    }
}

/* 
 * Free hashtable.. Returns number freed
 */ 
int free_hash_table(hashtable_t* hashtable){
    int ctr = 0; 
	int chain_len; 
    for (int i = 0; i < hashtable->size; i++){
        entry_t* current = hashtable->table[i]; 
        if (current){
        	chain_len = free_entry_chain(current); 
        	// If we want to view collision efficiency... 
        	// printf("Length of chain: %d\n", chain_len); 
        	ctr += chain_len;
        }
    }
    free(hashtable->table); 
    free(hashtable); 
    return ctr;
}

