#!/bin/bash

link=$1
size=(s5 1x 2x 3x)
gsin=(6.3 12.6 25.2 37.8)
gsout=(0.0 0.0 0.0 0.0)
wsin=(4.75 9.5 19.0 28.5)
wsout=(0.00003 0.00006 0.00012 0.00018)
ghin=(4.75 9.5 19.0 28.5)
ghout=(0.0 0.0 0.0 0.0)
tgin=(0.0 0.0 0.0 0.0)
tgout=(5.5 11.0 22.0 33.0)
whin=(3.15 6.3 12.6)
whout=(0.00002 0.00005 0.00010)
shin=(1.5 3.1 6.1)
shout=(1.5 3.1 6.1)

i=0
j=1
#rm $link/*size
#i=0
#for (( i=0; i<4; i++ )) ; do
		files=(`echo $link | grep grepspark `)
		for file in ${files[@]} ; do
#			echo ${file}_temp
			while read line ; do
				echo ${gsin[$i]},${gsin[$j]},${gsout[$j]},${gsout[$i]},$line >> ${file}_temp
			done < $file
		done

		files=(`echo $link | grep wcspark`)
		for file in ${files[@]} ; do
#			echo ${file}_temp
			while read line ; do
				echo ${wsin[$i]},${wsin[$j]},${wsout[$j]},${wsout[$i]},$line >> ${file}_temp
			done < $file
		done

		files=(`echo $link | grep teragen`)
                for file in ${files[@]} ; do
#                        echo ${file}_temp
                        while read line ; do
                                echo ${tgin[$i]},${tgin[$j]},${tgout[$j]},${tgout[$i]},$line >> ${file}_temp
                        done < $file
                done
		files=(`echo $link | grep grephadoop`)
                for file in ${files[@]} ; do
#                        echo ${file}_temp
                        while read line ; do
                                echo ${ghin[$i]},${ghin[$j]},${ghout[$j]},${ghout[$i]},$line >> ${file}_temp
                        done < $file
                done


save=Temp

if [ ! -d $save ]; then
	mkdir $save

fi


mv ${link}_temp $save


