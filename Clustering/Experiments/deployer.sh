#! /bin/bash

# declare array 
declare -a arr=("rolani@lab2-2.cs.mcgill.ca" "rolani@lab2-3.cs.mcgill.ca" "rolani@lab2-4.cs.mcgill.ca" 
"rolani@lab2-5.cs.mcgill.ca" "rolani@lab2-6.cs.mcgill.ca" "rolani@lab2-7.cs.mcgill.ca"
"rolani@lab2-8.cs.mcgill.ca" "rolani@lab2-9.cs.mcgill.ca" "rolani@lab2-10.cs.mcgill.ca" 
"rolani@lab2-11.cs.mcgill.ca" "rolani@lab2-12.cs.mcgill.ca" "rolani@lab2-13.cs.mcgill.ca"
"rolani@lab2-17.cs.mcgill.ca" "rolani@lab2-18.cs.mcgill.ca" "rolani@lab2-19.cs.mcgill.ca" 
"rolani@lab2-20.cs.mcgill.ca" "rolani@lab2-21.cs.mcgill.ca" "rolani@lab2-22.cs.mcgill.ca" 
"rolani@lab2-23.cs.mcgill.ca" "rolani@lab2-25.cs.mcgill.ca" "rolani@lab2-26.cs.mcgill.ca" 
"rolani@lab2-27.cs.mcgill.ca" "rolani@lab2-29.cs.mcgill.ca" "rolani@lab2-30.cs.mcgill.ca"
"rolani@lab2-31.cs.mcgill.ca"  "rolani@lab2-32.cs.mcgill.ca" "rolani@lab2-34.cs.mcgill.ca" 
"rolani@lab2-39.cs.mcgill.ca"  "rolani@lab2-40.cs.mcgill.ca"  "rolani@lab2-42.cs.mcgill.ca"
"rolani@lab2-44.cs.mcgill.ca"  "rolani@lab2-48.cs.mcgill.ca" "rolani@lab2-50.cs.mcgill.ca" 
"rolani@lab2-51.cs.mcgill.ca")

#declare -a arr=("rolani@lab2-2.cs.mcgill.ca" "rolani@lab2-3.cs.mcgill.ca" "rolani@lab2-4.cs.mcgill.ca")

# array size
echo "${#arr[@]}"



#loop through array 
for i in "${arr[@]}"
    do
       sys_name=${i:7}
	   name=${sys_name%%.*}
	   echo $name
       #echo "Start_time, End_time, Duration" >> $name
	   ssh $i "cd /home/2016/rolani/new_experiments; ./start.sh $name >> log &"
    done