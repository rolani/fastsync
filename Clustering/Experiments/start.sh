#! /bin/bash

#printf "Start_time, End_time, Duration, Memory_Usage_Percent, CPU_Usage_Percent \n"

#echo "Start_time, End_time, Duration, Memory_Usage_Percent, CPU_Usage_Percent" >> temp_out

c=0
while [ $c -lt 3000 ];
do

   declare -a arr=("task1.py" "task2.py" "task3.py" "task4.py")
	for i in "${arr[@]}"
	do
		python $i $1
	done 
c=$[$c+1]	
done



