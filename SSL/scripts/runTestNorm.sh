#!/bin/bash


data=(`cat runtest.conf`)

if [ $# -lt 3 ]; then
	echo Usage: $0 ModelFolder TimeCreateTest TestName
	exit
fi

cur=`pwd`
dataPath=${cur}/TEST/ARFF

#modelPath=${cur}/Result/
#modelPath=${PWD}/Result/20170731_Profiling_SOLO_1x2x3xs5_CPs5_adding_Filtered_collect_7newAttrs
#modelPath=${PWD}/Result/20170731_PLANB_RWSR_Profile_SOLO_CPs5_collect_TRAIN
#modelPath=${PWD}/Result/20170731_Profiling_SOLO_1x2x3xs5_CPs5_adding_Filtered_collect_7newAttrs
#modelPath=${PWD}/Result/20170729_Profile_SOLO_1x2x3xs5_TRAIN_addFiltered_collect
#modelPath=${PWD}/Result/20170731_Profile_SOLO1x2x3xs5_CP_ALL_Filter_collect
#modelPath=${PWD}/Result/20170731_Profile_SOLO1x2x3xs5_CP_s5_Filter_collect
#modelPath=${PWD}/Result/20170802_ModelI_test6_10_14_2_4_5_11_6_collect.arff
#modelPath=${PWD}/Result/ModelIO3_1_10_14_2_4_5_Profile_SOLO_targetgreater_CPs5_collect
#modelPath=${PWD}/Result/ModelIO2_1_10_14_2_4_5_Profile_SOLO_s5_CPs5_collect
#modelPath=${PWD}/Result/ModelIO1_1_10_14_2_4_5_Profile_SOLO_excludeds5_CPs5_collect
#modelPath=${PWD}/Result/ModelInew3_11_6_4_5_10_2_14_Profile_SOLO_target_greater_CPs5_collect
#modelPath=${PWD}/Result/ModelInew2_11_6_4_5_10_2_14_Profile_SOLO_s5_CPs5_collect
#modelPath=${PWD}/Result/MODELI_runAgain_collect.arff
#modelPath=${PWD}/Result/170804_Filtered_MLDATA_10_14_2_4_5_A
#modelPath=${PWD}/Result/170804_Filtered_MLDATA_1_10_14_2_4_5_11_6_T
#modelPath=${PWD}/Result/170804_Filtered_MLDATA_10_14_2_4_5_11_B
#modelPath=${PWD}/Result/170804_Filtered_MLDATA_10_14_2_4_5_6_D
#modelPath=${PWD}/Result/170804_Filtered_MLDATA_1_10_14_2_4_5_6_F
#modelPath=${PWD}/Result/170804_Filtered_MLDATA_1_10_14_2_4_5_11_C

echo `date`>> log_runTest
modelFolder=$1
echo $1  >> log_runTest

modelPath=${PWD}/Result/$modelFolder
#modelPath=${PWD}/Result/20170805_170805_ModelI5_11_6_4_5_10_2_14_Filtered_Converted_collect
#modelPath=${PWD}/Result/20170805_ModelInew1_11_6_4_5_10_2_14_Profile_SOLO_excludes5_CPs5_collect_T2
#modelPath=${PWD}/Result/20170803_ModelInew1_11_6_4_5_10_2_14_Profile_SOLO_excludes5_CPs5_collect_T2_2222.arff
#modelPath=${PWD}/Result/20170803_ModelInew1_11_6_4_5_10_2_14_Profile_SOLO_excludes5_CPs5_collect_T1_11111111.arff
#modelPath=${PWD}/Result/20170803_ModelInew1_11_6_4_5_10_2_14_Profile_SOLO_excludes5_CPs5_collect_T1T2_22.arff
#modelPath=${PWD}/Result/170804_ModelD_Profile_SOLO_excludes5_CPs5_170804_Converted_Filtered_collect.arff
#modelPath=${PWD}/Result/MODELI_runAgain_collect.arff
#modelPath=${PWD}/Result/20170804_ModelInew1_11_6_4_5_10_2_14_Profile_SOLO_excludes5_CPs5_170804_Converted_Filtered_collect
#modelPath=${PWD}/Result/ModelInew1_11_6_4_5_10_2_14_Profile_SOLO_excludes5_CPs5_collect
#modelPath=${PWD}/Result/Model_Test4
#modelPath=${PWD}/Result/20170802_Model_Test5_Filtered_MLDATA_10_14_2_4_5_6_collect.arff
#modelPath=${PWD}/Result/ModelI_test6_10_14_2_4_5_11_6_collect.arff

#modelPath=${PWD}/Result/20170802_ModelI_test4_10_14_2_4_5_size_collect_collect.arff

#modelPath=${PWD}/Result/20170802_Model_Test5_Filtered_MLDATA_10_14_2_4_5_6_collect.arff
#modelPath=${PWD}/Result/20170802_ModelI_test1_triednoSolos5_collect.arff/
#modelPath=${PWD}/Result/20170802_ModelI_test5_10_14_2_4_5_6_collect.arff
#modelPath=${PWD}/Result/20170802_ModelI_test4_10_14_2_4_5_11_collect.arff
#modelPath=${PWD}/Result/10_14_2_4_5_ModelI_test
#modelPath=${PWD}/Result/MODELI_runAgain_collect.arff
#modelPath=${PWD}/Result/20170802_ModelI_test3_Profile_SOLO_CPs5_1_10_14_2_4_5_11_collect.arff
#modelPath=${PWD}/Result/ModelI_test2_Profile_SOLO_CPs5_1_10_14_2_4_5_6_collect.arff
#modelPath=${PWD}/Result/ModelI_test1_Profile_SOLO_CPs5_10_14_2_4_5_collect.arff
#modelPath=${PWD}/Result/MODELI4_Profile_SOLO_CP_size_except_itsefl_addedFilter11_6_4_5_10_2_14_8_collect
#modelPath=${PWD}/Result/MODELI5_Profile_SOLO_CP_size_except_itsefl_addedFilter11_6_4_5_10_2_14_collect
#modelPath=${PWD}/Result/170801_modelI_rofile_SOLO_CPs5_addingFilter_11_12_13_5_6_8_9_collect
#modelPath=${PWD}/Result/20170731_Profile_SOLO1x2x3xs5_CP_s5_Filter_collect
#modelPath=${PWD}/Result/20170729_Profile_SOLO_1x2x3xs5_TRAIN
#modelPath=${PWD}/Result/20170729_Profile_SOLO_1x2x3xs5_TRAIN
#modelPath=${PWD}/Result/Origninal_SOLO_NEWAPPS_X3_collect
#modelPath=${PWD}/Result/Origninal_SOLO_NEWAPPS_X3_collect/
#modelPath=${PWD}/Result/170724_Origninal_SOLO_NEWAPPS_S3profile_newappsS2_proflie_ADD_s3
#modelPath=/home/gorae/NewVersion_170206/Result/20170716_smallI1
#modelPath=/home/gorae/NewVersion_170206/Result/20170716_smallI1_noX1/
#modelPath=/home/gorae/NewVersion_170206/Result/20170716_smallI2_noX1/
#modelPath=/home/gorae/NewVersion_170206/Result/20170716_smallI2/
#modelPath=/home/gorae/NewVersion_170206/Result/Old_20170712/20170711_noMap_new/
#modelPath=/home/gorae/NewVersion_170206/Result/

methods="ANN Gauss REPTree SMO RandomTree"
#methods="RandomTree"
#c=storage_170228
#c=C3C4
#c=DATA_BLUEa

#c=WLS5_1X_SOLOS5_2x_3x_LARGESCALE_new3_collect_0801
#c=smallInput2
#c=DATA_TEST_s5_target1x_newAttr_FM
#c=DATA_TEST_s5_target1x_newAttr_FM
#c=WL_s31x
#c=WL_TEST_SOLOS5_collect
#c=WLs5_1x_SOLOS5_collect
#c=ONLY_LARGE_SCENARIO_collect
#c=WLs5_1x_SOLOS5_LARSCALE_new1500_collect
#c=PLANB_LARGESCALE_RWSR
#c=170801_LARGESCALE_ONLY_collect_new3/

#for c in 170803_ConvertedLargeScenario_newAttr_size_collect 170803_DATA_TEST_11_3_13_12_8_7_6_9_SOLOs5_2x3x_only_collect 170803_TEST_11_3_13_12_8_7_6_9_SOLOS5_2x_3x_LARGESCALE_collect; do
#for c in 170803_ConvertedLargeScenario_newAttr_size_collect 170803_TEST_3_13_12_7_9_1_8_SOLO_s5_2x_3x_LARGESCALE_collect 170803_DATA_TEST_3_13_12_7_9_1_8_SOLO_s5_2x_3x_ONLY_collect 170803_DATA_TEST_3_13_12_7_9_8_SOLO_s5_2x_3x_ONLY_collect 170803_TEST_3_13_12_7_9_8_SOLO_s5_2x_3x_LARGESCALE_collect; do
#for c in WLS5_1X_SOLOS5_2x_3x_LARGESCALE_new3_collect_0801 170801_LARGESCALE_ONLY_collect_new3; do
#c=WLS5_1X_SOLOS5_2x_3x_LARGESCALE_new3_collect_0801
#c=WL_3_13_12_7_9_1_8_s51x_SOLOS5_2x3x_LARGESCALEnew3_collect
#c=WL3_13_12_7_9_8_SOLOS5_2x_3x_LARGESCALE_new3_collect
#c=ModelI_test1_Profile_SOLO_CPs5_10_14_2_4_5_collect.arff
#c=WL_3_13_12_7_9_1_s51x_SOLOS5_2x_3x_LARGESCALEnew3_collect
#c=170801_LARGESCALE_ONLY_collect_new3
#c=PLANB_RWRS_LARGESCALEONLY_170801_collect
#c=PLANB_RWRS_WLs5_1x_attrs_LARGESCALE_170801_collect
#c=PLANB_LARGESCALE_RWSR_collect_170801
#c=LARGESCALEONLY_170801
#c=WLS5_1X_SOLOs5_LargeScaleNEW_170801_collect
#c=PLANB_WLs5_1x_attrs_SOLO_Largescale_collect_RWRS
#c=PLANB_LARGEINPUT_ALL_attrs_collect
#c=WLs5_1x_SOLO_LARGESCALE_collect
#c=LARGESCALE_FM_new_1500_0731_collect
#c=WLs5_1x_SOLOS5_LargeScale_collect
#c=WLs5_1x_SOLOS5_CP5_collect
#c=s3_1x2x3x
#c=s3_1x2x3x
#c=SOLO3x_Target1x2x
#c=newcfg_newapps_3inf_noMap
#c=newcfg_newapps_noMap
#c=newconfig
#c=WL11_3_13_12_8_7_6_9_Complete
#c=WL11_3_13_12_8_7_6_9_Complete_4T_format
#timeTrain=20161207
#timeTrain=20170403
#timeTrain=20170525
#for c in 170804_WL_3_13_12_7_9_8_ConvertedLargeScenario_collect_solo_s5_target_1x_collect ; do
#for c in 170804_ConvertedLargeScenario_newAttr_size_collect ; do
#for c in 170804_WL_3_13_12_7_9_8_ConvertedLargeScenario_collect; do


#170804_ConvertedLargeScenario_newAttr_size_collectfor c in 170804_WL_3_13_12_7_9_8_ConvertedLargeScenario_collect_solo_s5_target_1x_collect WL3_13_12_7_9_8_SOLOS5_2x_3x_LARGESCALE_new3_collect; do
#for c in 170804_ConvertedLargeScenario_newAttr_size_collect_T2_2222 170804_WL_3_13_12_7_9_8_ConvertedLargeScenario_collect_solo_s5_target_1x_collect_T2_2222; do
#for c in 170804_ConvertedLargeScenario_newAttr_size_collect_T1T2_22 170804_WL_3_13_12_7_9_8_ConvertedLargeScenario_collect_solo_s5_target_1x_collect_T1T2_22 ; do
#timeTrain=20170729
#timeTrain=20170712
#timeTrain=20170626
#timeTrain=20170327
#timeTrain=20170312
#timeTrain=20170310


#timeTest=20170731
timeTest=$2
c=$3

echo $2_$3 >> log_runTest
#timeTest=20170801
#timeTest=20170731
#timeTest=20170722
#timeTest=20170712
#timeTest=20170626
#timeTest=20170525
#timeTest=20170310
#timeTest=20170403
#timeTest=20170327
#timeTest=20170310
#timeTest=20161212

for app in ${data[@]}; do
	echo $app

done
linkPath=${dataPath}/${timeTest}_$c
echo $linkPath
for app in ${data[@]}; do

	appname=`ls ${linkPath}| grep ${app} `
	echo $appname 
	linkTest=$linkPath/${appname}
	echo $linkTest >> log_runTest

	ls $modelPath | grep $app | grep -v "result" | grep -v "norm" |  while read -r modelFolder; do
	#ls $modelPath | grep $app | grep -v "result" |  while read -r modelFolder; do
		echo $modelFolder 
		link=$modelPath/$modelFolder/
		for method in $methods; do
			linkMethod=$link/$method/

			echo $linkMethod 
			t=`ls $linkMethod|grep "model" | grep -v "norm" |  grep -v "result"`
			#echo $t
			model=$linkMethod/${t}
			cd JavaCodeNormalizeTeragen
			rm *.class
			javac Test.java  >> log_runTest 2>&1
			java Test $model $linkTest $linkMethod/${app}_${t}_result_${timeTest}_$c >> log_runTest 2>&1
			cd ..


#			if [[ $method == "REPTree" ]] || [[ $method == "RandomTree" ]]; then
#				linkMethod1=$link/${method}_BAGGING/
#				echo $linkMethod1
#				t=`ls $linkMethod1 | grep "model" | grep -v "result"`
#				model=${linkMethod1}/${t}
#				echo $model
#				cd Test_bagging
#					rm *.class
#					javac TestBagging.java
#					java TestBagging $model $linkTest ${linkMethod1}/${app}_${t}_result_${timeTest}
#				cd ..


#			fi
		done
	done
done
