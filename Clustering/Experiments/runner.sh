#! /bin/bash

declare -a arr=("task1.py" "task2.py" "task3.py" "task4.py")
file_out="temp_out.txt"
for i in "${arr[@]}"
do
	#start=$(date +%s.%N)
	#start=$(date +%s)
	python $i $file_out
	#end=$(date +%s.%N)
	#end=$(date +%s)  
	#runtime=$(python -c "print(${end} - ${start})")
	#printf "%09.9f,%09.9f,%02.5f\n"  $start $end $runtime
done
