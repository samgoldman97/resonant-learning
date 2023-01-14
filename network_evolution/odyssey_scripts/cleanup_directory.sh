#!/bin/sh

for targs in `seq 1 1 7`
do
    
    python3 cleanup_files_passed.py long_oscil_multiplex_"$targs"_target.p *oscil*"$targs"_target*
    python3 cleanup_files_passed.py long_fixed_multiplex_"$targs"_target_free.p *fixed_start_free*"$targs"_target*  \
	    *fixed*"$targs"_target_free*.p
    python3 cleanup_files_passed.py long_fixed_multiplex_"$targs"_target_const.p *fixed_start_const*"$targs"_target*  \
	    *fixed*"$targs"_target_const*.p    
done

