#! /bin/bash

# Helper function to test the odyssey scripts on home computer

DATE=`date '+%y-%m-%d-%H-%M-%S'`

for i in `seq 1 4`;
do		
    python3 main.py "$i$DATE"&
done



