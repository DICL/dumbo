#!/bin/bash

if [ $# -lt 3 ]; then
	echo Usage: $0 time character data/application name
	exit
fi

data=$3 # can be cg, ...


if [[ ! -d $path ]]; then
	mkdir $path
fi

time=$1
c=$2
folder=${time}_${c}
cur=`pwd`
path=${cur}/Result/
TrainPath=${cur}/TRAIN/ARFF/$folder
echo $TrainPath

for app in ${data[@]}; do
	echo "Run TRain $app"

	app_name=`ls $TrainPath | grep $data`
	trainResult=$path/${app_name}
	if [[ ! -d $trainResult ]]; then
		mkdir $trainResult
	fi
	
	train=$TrainPath/${app_name}
	echo $train
	cd JavaCodeNormalizeCG
		rm -rf *.class
		javac Train.java
		java Train $train $trainResult
	cd ..


	#cd Test_bagging

	#	rm -f *.class
	#	javac TrainBagging.java
	#	java TrainBagging $train $trainResult

	#cd ..

done

