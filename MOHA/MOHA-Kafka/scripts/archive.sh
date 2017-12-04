#!/bin/bash
# script for archiving input data files and generating a set of data-bundles

if [ $# -ne 3 ]; then
  echo "[Usage]: $0 <Input-Directory> <Output-Directory> <Bundle-Unit>"
  exit 1
fi

input_dir=$1
output_dir=$2
bundle_unit=$3

mkdir -p $output_dir # Create the output directory if needed

# Move into the data directory
cd $input_dir
input_files=(*)
totalNumOfFiles=${#input_files[*]} # Get the total number of data files (directories)

echo "# Processing $totalNumOfFiles files in $input_dir"
bundle_group_id=1

for (( i = 0; i < $totalNumOfFiles; )); 
do
	end_of_bundle=$((i + bundle_unit))
	if [ $end_of_bundle -gt $totalNumOfFiles ]; then
		end_of_bundle=$totalNumOfFiles
	fi

	echo
	echo "# Bundle Group $bundle_group_id: $(($end_of_bundle - $i)) data elements"
	bundle_command="tar zcf data-bundle_$bundle_group_id.tar.gz "
	for (( j = $i; j < $end_of_bundle; j++ ));
	do
		bundle_command="$bundle_command ${input_files[$j]}"
	done

	echo "# executing \"$bundle_command\""
	eval $bundle_command

	i=$end_of_bundle
	bundle_group_id=$((bundle_group_id + 1))
done

# Move the data bundle files
output_dir_name=$(basename $output_dir)
mv data-bundle*.tar.gz ../$output_dir/

echo
echo "# Completed generating $((bundle_group_id -1)) data-bundle groups"

# Move out from the data directory
cd ..
