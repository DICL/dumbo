#!/bin/sh
if [ $# -eq 3 ]; then

# Check if the executable file exists
if [ -f ./vina ]
then
	echo "# Autodock Vina executable is successfully found"
else
	echo "# [ERROR] Autodock Vina executable is not found"
	exit 0
fi

ligand_file=$1  # ligand file
protein_folder=$2 # protein folder
coordinates_file=$3 # coordinates file

echo "Coordinates File = $coordinates_file";

# extract only file/directory names of protein and ligand
ligand_name=`basename $ligand_file .pdbqt`
protein_name=`basename $protein_folder`

result_dir=docking_results

export AUTODOCK_UTI=$PWD
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH

ulimit -s unlimited

center_x=0
center_y=0
center_z=0
num=`cat $coordinates_file | grep $protein_name | wc -l`
echo "# The number of site coordinates: $num"
if [ $num -eq 1 ]; then
	center_x=`cat $coordinates_file | grep $protein_name | cut -f 2`
	center_y=`cat $coordinates_file | grep $protein_name | cut -f 3`
	center_z=`cat $coordinates_file | grep $protein_name | cut -f 4`
fi

echo "# PDB Coordinates: ($center_x, $center_y, $center_z)"

chmod +x ./vina

mkdir -p $result_dir

echo "# Drug Repositioning Simulation starts: $ligand_name + $protein_name"
for p in $protein_folder/*.pdbqt; do
	echo "# docking of protein $p"
	output="$ligand_name+$protein_name+`basename $p .pdbqt`"

	# Autodock Vina docking command
	./vina --receptor $p --ligand $ligand_file --out $output.pdbqt --log $output.log --center_x $center_x --center_y $center_y --center_z $center_z --size_x 20 --size_y 20 --size_z 20 --cpu 1 --seed 1
	
	if grep "Writing output ... done." $output.log; then
  		echo "# docking is successfully completed"
		mv $output.pdbqt $result_dir/
		mv $output.log $result_dir/
	else
		echo "# docking has failed"
	fi
done


#echo "Archiving output files.."
#rm -rf $protein
#mv $output $protein
#tar cvf $ligand'_'$protein.tar $protein

echo "# Drug Repositioning Simulation is completed"

else
	echo "[Usage] autodock_vina.sh <ligand_file> <protein_folder> <coordinates_file>"
fi
