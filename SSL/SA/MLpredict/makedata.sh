#!/bin/bash
link=$1
linksave=$2
ls $link | while read -r name; do

	if [[ $name != "makedata"* ]] && [[ $name != "Data" ]] ; then
		n=`awk -F\, '{print NF; exit}' ${link}/$name`
		n=$(( $n - 1 ))
		echo @relation $name > ${linksave}/${name}.arff
		for i in $(seq 1 1 $n) 
		do	
			echo @attribute attr${i} real >> ${linksave}/${name}.arff
		done
		echo @attribute class real >> ${linksave}/${name}.arff
		echo @data >> ${linksave}/${name}.arff
		cat  ${link}/$name >> ${linksave}/${name}.arff
	fi			
done

