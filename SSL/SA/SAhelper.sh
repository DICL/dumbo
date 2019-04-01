#!/bin/bash

pid=`ps ax | grep -v grep | grep SA_Exec | awk '{print $1}'`

kill -stop $pid

linkData=`pwd`
link=`pwd`
num=`ls ${linkData}/*.sample | wc -l`

cd ${linkData}
files=(`ls *.sample`)
#cd $link
cd MLpredict
#cd HelpST_Individual

# MAKE 40 ATTRIBUTE FILE
for file in ${files[@]} ; do


	if [[ $file == *"lammps"* ]] || [[ $file == *"namd"* ]] || [[ $file == *"cg"* ]] || [[ $file == *"132"*  ]]; then
	
#        	./SelectParser.sh $linkData/$file > Temp/${file}_temp
        	./SelectParser40.sh $linkData/$file > Temp/${file}_temp

	else 
#        	./SelectParser.sh $linkData/$file > Temp/${file}_temp

#		mv $linkData/temp $linkData/$file
#		./SelectParser68.sh $linkData/${file} > 68Attrs/${file}

		./CTDS appdat_1x appdat_s5 < $linkData/$file > $linkData/temp
		./SelectParser68.sh $linkData/temp > 68Attrs/${file}
		./addsize.sh 68Attrs/${file}

	fi
#	rm $file
#	mv ${file}_temp $file
done
# create THE FILE FORMAT
./makedata.sh Temp Temp_FM


files=(`ls Temp_FM`)
for ((i=0; i<$num; i++)); do
	files[$i]="Temp_FM/"${files[$i]}
done

#echo ${files[@]}


rm -r *.class 2> /dev/null
javac Help.java 2> /dev/null
#java Help $num ${files[@]} > $linkData/Model_result 2> /dev/null
java Help $num ${files[@]} > $linkData/Model_result #2> /dev/null

#rm Temp/* Temp_FM/*
cd ..

kill -cont $pid

#rm Temp/* Temp_FM/*
