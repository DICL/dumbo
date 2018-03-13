#! /bin/bash




# 기본파일 설정
CONFIG_FILE='lustre-default-config.xml'
CONFIG_FILE_NAME='./'$CONFIG_FILE
CONFIG_FILE_PATH='/var/lib/ambari-server/resources/stacks/HDP/2.6/services/LUSTREMGMTSERVICES/configuration'

ADVENCE_CONFIG_FILE='lustrefs-config-env.xml'
ADVENCE_CONFIG_FILE_NAME='./'$ADVENCE_CONFIG_FILE
ADVENCE_CONFIG_FILE_PATH='/var/lib/ambari-server/resources/stacks/HDP/2.6/services/LUSTREMGMTSERVICES/configuration'


THEME_FILE='theme.json'
THEME_FILE_NAME='./'$THEME_FILE
THEME_FILE_PATH='/var/lib/ambari-server/resources/stacks/HDP/2.6/services/LUSTREMGMTSERVICES/themes'

CONF_FILE='./ambari_config.xml'


MDS=();
MDS_MOUNT=();
MDS_FSNAME=();

OSS=();
OSS_DISKS=();
OSS_MOUNT=();

Client=();


MDS_NETWORK=();OSS_NETWORK=();CLIENT_NETWORK=(); # ip addr show 결과문 저장
MDS_DF=();OSS_DF=();CLIENT_DF=(); # df 결과문 저장
MDS_size=();OSS_size=();

# # conf 파일 읽어오기
# while read -r heder conf1 conf2 conf3
# do 
# 	# 주석문 제외
# 	[[ ${heder} =~ \#.* ]] && continue
# 	# MDS OSS Client 구분
# 	if [[ (${heder} =~ "[MDS_SERVERS]") ]]; then
# 		is_type='MDS'
# 		continue;
# 	fi
# 	if [[ (${heder} =~ "[OSS_SERVERS]") ]]; then
# 		is_type='OSS'
# 		continue;
# 	fi
# 	if [[ (${heder} =~ "[Client_SERVER]") ]]; then
# 		is_type='CLIENT'
# 		continue;
# 	fi

# 	if [ ${is_type} == "MDS" ] ; 
# 	then
# 		if [[ ($heder != "") ]]; then
# 			MDS+=(${heder});
# 			MDS_MOUNT+=(${conf1});
# 			MDS_FSNAME+=(${conf2});
# 		fi
# 	else
# 		if [ ${is_type} == "OSS" ] ; 
# 		then
# 			if [[ ($heder != "") ]]; then
# 				OSS+=(${heder});
# 				OSS_DISKS+=(${conf1});
# 				OSS_MOUNT+=(${conf2});
# 			fi
# 		else
# 			if [ ${is_type} == "CLIENT" ] ; 
# 			then
# 				if [[ ($heder != "") ]]; then
# 					Client+=(${heder});
# 				fi
# 			fi
# 		fi
# 	fi
# done < $CONF_FILE

MDS=(`xmllint --xpath '//MDS_SERVERS/host/text()' ${CONF_FILE}`)
MDS_MOUNT=`xmllint --xpath '//MDS_SERVERS/mountpoint/text()' ${CONF_FILE}`
MDS_FSNAME=`xmllint --xpath '//MDS_SERVERS/fsname/text()' ${CONF_FILE}`

declare -a OSS=( )
declare -a OSS_DISKS=( )
ossCount=`xmllint --xpath 'count(//OSS_SERVERS/server)' ${CONF_FILE}`

for (( i=1; i <= $ossCount; i++ )); do
    host=`xmllint --xpath '//OSS_SERVERS/server['${i}']/host/text()' ${CONF_FILE}`;
    OSS+=(${host});
    disk=`xmllint --xpath '//OSS_SERVERS/server['${i}']/disk_num/text()' ${CONF_FILE}`;
    OSS_DISKS+=(${disk});
done
Client=`xmllint --xpath '//Client_SERVER/mountpoint/text()' ${CONF_FILE}`
echo 'read conf file.';


# MDS_HOST_NUM=${#MDS[@]};
# OSS_HOST_NUM=${#OSS[@]};
# CLIENT_HOST_NUM=${#CLIENT[@]};

echo "${MDS[@]}";
echo "${OSS[@]}";
echo "${Client[@]}";


for HOST in "${MDS[@]}";
do
	IFS=$'\n';i=0;
	ARR=(`ssh -o StrictHostKeyChecking=no root@${HOST} "ip addr show | grep '^[0-9]: [0-9|a-z]*'" `);
	unset IFS;
	TEMP=();
	for LINE in "${ARR[@]}"; do
		WORDS=(${LINE});
		TEMP2=${WORDS[1]:0:(-1)}
		TEMP+="${TEMP2}|";
	done
	MDS_NETWORK+=("${TEMP}");


	IFS=$'\n';i=0;
	ARR=(`ssh -o StrictHostKeyChecking=no root@${HOST} "fdisk -l"`);
	unset IFS;
	TEMP=();
	TEMP2=();
	for LINE in "${ARR[@]}"; do
		WORDS=(${LINE});
		if [[ ("$WORDS[0]" =~ "/dev") ]]; then
			echo "${WORDS[0]}"
			TEMP+="${WORDS[0]}|";
			ARR2=(`ssh -o StrictHostKeyChecking=no root@${HOST} "fdisk -l ${WORDS[0]} | grep \"Disk\"" `);
			if [[ ("${ARR2[3]}" =~ "MB") ]] ;
				then
					TEMP2+="$((ARR2[2]/1024))|";
				else
					# TEMP2+="${ARR2[2]}|";
					TEMP3=$(echo "${ARR2[2]}"| awk -F '.' '{print $1}')
					TEMP2+="${TEMP3}|";
			fi
		fi
	done
	MDS_DF+=("${TEMP}");
	MDS_size+=("${TEMP2}");

done



for HOST in "${OSS[@]}";
do
	IFS=$'\n';i=0;
	ARR=(`ssh -o StrictHostKeyChecking=no root@${HOST} "ip addr show | grep '^[0-9]: [0-9|a-z]*'" `);
	unset IFS;
	TEMP=();
	for LINE in "${ARR[@]}"; do
		WORDS=(${LINE});
		TEMP2=${WORDS[1]:0:(-1)}
		TEMP+="${TEMP2}|";
	done
	OSS_NETWORK+=("${TEMP}");


	IFS=$'\n';i=0;
	ARR=(`ssh -o StrictHostKeyChecking=no root@${HOST}  "fdisk -l"`);
	unset IFS;
	TEMP=();
	TEMP2=();
	for LINE in "${ARR[@]}"; do
		WORDS=(${LINE});
		# echo $WORDS
		if [[ ("$WORDS[0]" =~ "/dev") ]]; then
			echo "${WORDS[0]}"
			TEMP+="${WORDS[0]}|";
			ARR2=(`ssh -o StrictHostKeyChecking=no root@${HOST} "fdisk -l ${WORDS[0]} | grep \"Disk\"" `);
			if [[ ("${ARR2[3]}" =~ "MB") ]] ;
				then
					TEMP2+="$((ARR2[2]/1024))|";
				else
					# TEMP2+="${ARR2[2]}|";
					TEMP3=$(echo "${ARR2[2]}"| awk -F '.' '{print $1}')
					TEMP2+="${TEMP3}|";
			fi
			
		fi
	done
	OSS_DF+=("${TEMP}");
	OSS_size+=("${TEMP2}");

done
# echo "$MDS_size[@]";
# echo "$MDS_DF[@]";
# echo "$OSS_size[@]";
# echo "$OSS_DF[@]";
echo 'get server information';




# 
# 파일생성부분
# 

echo '<?xml version="1.0"?>' > 													$CONFIG_FILE_NAME;
echo '<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>' >> 			$CONFIG_FILE_NAME;
echo '<configuration supports_final="false">' >> 								$CONFIG_FILE_NAME;

# mds 드라이브 선택부분
i=0;
for HOST in "${MDS[@]}"; do
echo '    <property>' >> 														$CONFIG_FILE_NAME;
echo '        <name>device_mds-'${HOST}'</name>' >> 							$CONFIG_FILE_NAME;
echo '        <display-name>'${HOST}' MDT Server Device</display-name>' >> 			$CONFIG_FILE_NAME;
echo '        <value/>' >> 														$CONFIG_FILE_NAME;
echo '        <description>'${HOST}' MDT Device Select</description>' >>			$CONFIG_FILE_NAME;
echo '        <value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '            <type>value-list</type>' >> 									$CONFIG_FILE_NAME;
echo '            <entries>' >> 												$CONFIG_FILE_NAME;

IFS=$'|';
HOST_DEVICE=(${MDS_DF[${i}]});
unset IFS;
for LINE in "${HOST_DEVICE[@]}"; do
	echo '            <entry><value>'${LINE}'</value></entry>' >> 				$CONFIG_FILE_NAME;
done

echo '            </entries>' >> 												$CONFIG_FILE_NAME;
echo '            <selection-cardinality>2+</selection-cardinality>' >> 		$CONFIG_FILE_NAME;
echo '        </value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '    </property>' >> 														$CONFIG_FILE_NAME;

# # mds 드라이브 사이즈
# echo '    <property>' >> 														$CONFIG_FILE_NAME;
# echo '        <name>device_mds_size-'${HOST}'</name>' >> 						$CONFIG_FILE_NAME;
# echo '        <display-name>'${HOST}' MDT Device Size</display-name>' >> 			$CONFIG_FILE_NAME;
# echo '        <value>10</value>' >> 												$CONFIG_FILE_NAME;
# echo '        <description>Size</description>' >> 								$CONFIG_FILE_NAME;
# echo '        <value-attributes>' >> 											$CONFIG_FILE_NAME;
# echo '            <type>int</type>' >> 											$CONFIG_FILE_NAME;
# echo '            <minimum>1</minimum>' >> 										$CONFIG_FILE_NAME;
# echo '            <maximum>10</maximum>' >> 									$CONFIG_FILE_NAME;
# echo '            <unit>GB</unit>' >> 											$CONFIG_FILE_NAME;
# echo '            <increment-step>1</increment-step>' >> 						$CONFIG_FILE_NAME;
# echo '        </value-attributes>' >> 											$CONFIG_FILE_NAME;
# echo '    </property>' >> 														$CONFIG_FILE_NAME;


j=0;
IFS=$'|';
HOST_DEVICE_SIZE=(${MDS_size[${i}]});
unset IFS;
for LINE in "${HOST_DEVICE_SIZE[@]}"; do
# mds 드라이브 사이즈
echo '    <property>' >> 														$CONFIG_FILE_NAME;
echo '        <name>device_mds_size-'${HOST}'-'${HOST_DEVICE[${j}]////}'</name>' >> 						$CONFIG_FILE_NAME;
echo '        <display-name>'${HOST}' MDT Device Size ('${HOST_DEVICE[${j}]}')</display-name>' >> 			$CONFIG_FILE_NAME;
echo '        <value>'${LINE}'</value>' >> 												$CONFIG_FILE_NAME;
echo '        <description>Size</description>' >> 								$CONFIG_FILE_NAME;
echo '        <value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '            <type>int</type>' >> 											$CONFIG_FILE_NAME;
echo '            <minimum>1</minimum>' >> 										$CONFIG_FILE_NAME;
echo '            <maximum>'${LINE}'</maximum>' >> 									$CONFIG_FILE_NAME;
echo '            <unit>GB</unit>' >> 											$CONFIG_FILE_NAME;
echo '            <increment-step>0.1</increment-step>' >> 						$CONFIG_FILE_NAME;
echo '        </value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '    </property>' >> 														$CONFIG_FILE_NAME;
j=$((j+1));
done


# mds 네트워크 선택부분
echo '    <property>' >> 														$CONFIG_FILE_NAME;
echo '        <name>network_mds-'${HOST}'</name>' >> 							$CONFIG_FILE_NAME;
echo '        <display-name>'${HOST}' Server Network</display-name>' >> 		$CONFIG_FILE_NAME;
echo '        <value/>' >> 														$CONFIG_FILE_NAME;
echo '        <description>'${HOST}' Network Select</description>' >>			$CONFIG_FILE_NAME;
echo '        <value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '            <type>value-list</type>' >> 									$CONFIG_FILE_NAME;
echo '            <entries>' >> 												$CONFIG_FILE_NAME;

IFS=$'|';
HOST_DEVICE=(${MDS_NETWORK[${i}]});
unset IFS;
for LINE in "${HOST_DEVICE[@]}"; do
	echo '            <entry><value>tcp('${LINE}')</value><label>'${LINE}'</label></entry>' >> 				$CONFIG_FILE_NAME;
done

echo '            </entries>' >> 												$CONFIG_FILE_NAME;
echo '            <selection-cardinality>2+</selection-cardinality>' >> 		$CONFIG_FILE_NAME;
echo '        </value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '    </property>' >> 														$CONFIG_FILE_NAME;

i=$((i+1));
done



# oss 드라이브 선택부분
i=0;total=0;
for HOST in "${OSS[@]}"; do
	j=0; LIMIT=${OSS_DISKS[$i]};
	while (( j < LIMIT )); do

		# oss 드라이브 선택부분
		echo '    <property>' >> 															$CONFIG_FILE_NAME;
		echo '        <name>device_oss'$((j+1))'-'${HOST}'</name>' >> 						$CONFIG_FILE_NAME;
		echo '        <display-name>'${HOST}' OST'$((total+1))' Server Device </display-name>' >> 	$CONFIG_FILE_NAME;
		echo '        <value/>' >> 															$CONFIG_FILE_NAME;
		echo '        <description>'${HOST}' Device Select</description>' >>				$CONFIG_FILE_NAME;
		echo '        <value-attributes>' >> 												$CONFIG_FILE_NAME;
		echo '            <type>value-list</type>' >> 										$CONFIG_FILE_NAME;
		echo '            <entries>' >> 													$CONFIG_FILE_NAME;

		IFS=$'|';
		HOST_DEVICE=(${OSS_DF[${i}]});
		unset IFS;
		for LINE in "${HOST_DEVICE[@]}"; do
			echo '            <entry><value>'${LINE}'</value></entry>' >> 					$CONFIG_FILE_NAME;
		done

		echo '            </entries>' >> 													$CONFIG_FILE_NAME;
		echo '            <selection-cardinality>2+</selection-cardinality>' >> 			$CONFIG_FILE_NAME;
		echo '        </value-attributes>' >> 												$CONFIG_FILE_NAME;
		echo '    </property>' >> 															$CONFIG_FILE_NAME;


		# # 디스크 사이즈 부분
		# echo '    <property>' >> 														$CONFIG_FILE_NAME;
		# echo '        <name>device_oss_size'$((j+1))'-'${HOST}'</name>' >> 					$CONFIG_FILE_NAME;
		# echo '        <display-name>'${HOST}' OST'$((total+1))' Device Size</display-name>' >> $CONFIG_FILE_NAME;
		# echo '        <value>10</value>' >> 												$CONFIG_FILE_NAME;
		# echo '        <description>Size</description>' >> 								$CONFIG_FILE_NAME;
		# echo '        <value-attributes>' >> 											$CONFIG_FILE_NAME;
		# echo '            <type>int</type>' >> 											$CONFIG_FILE_NAME;
		# echo '            <minimum>1</minimum>' >> 										$CONFIG_FILE_NAME;
		# echo '            <maximum>10</maximum>' >> 									$CONFIG_FILE_NAME;
		# echo '            <unit>GB</unit>' >> 											$CONFIG_FILE_NAME;
		# echo '            <increment-step>1</increment-step>' >> 						$CONFIG_FILE_NAME;
		# echo '        </value-attributes>' >> 											$CONFIG_FILE_NAME;
		# echo '    </property>' >> 														$CONFIG_FILE_NAME;

		

		j=$((j+1));total=$((total+1));
	done

		k=0;
		IFS=$'|';
		HOST_DEVICE_SIZE=(${OSS_size[${k}]});
		unset IFS;
		for LINE in "${HOST_DEVICE_SIZE[@]}"; do
				# 디스크 사이즈 부분
		echo '    <property>' >> 														$CONFIG_FILE_NAME;
		echo '        <name>device_oss_size-'${HOST}'-'${HOST_DEVICE[${k}]////}'</name>' >> 					$CONFIG_FILE_NAME;
		echo '        <display-name>'${HOST}' OST'$((total+1))' Device Size ('${HOST_DEVICE[${k}]}')</display-name>' >> $CONFIG_FILE_NAME;
		echo '        <value>'${LINE}'</value>' >> 												$CONFIG_FILE_NAME;
		echo '        <description>Size</description>' >> 								$CONFIG_FILE_NAME;
		echo '        <value-attributes>' >> 											$CONFIG_FILE_NAME;
		echo '            <type>int</type>' >> 											$CONFIG_FILE_NAME;
		echo '            <minimum>1</minimum>' >> 										$CONFIG_FILE_NAME;
		echo '            <maximum>'${LINE}'</maximum>' >> 									$CONFIG_FILE_NAME;
		echo '            <unit>GB</unit>' >> 											$CONFIG_FILE_NAME;
		echo '            <increment-step>0.1</increment-step>' >> 						$CONFIG_FILE_NAME;
		echo '        </value-attributes>' >> 											$CONFIG_FILE_NAME;
		echo '    </property>' >> 														$CONFIG_FILE_NAME;

		k=$((k+1));
		done

echo '    <property>' >> 														$CONFIG_FILE_NAME;
echo '        <name>network_oss-'${HOST}'</name>' >> 							$CONFIG_FILE_NAME;
echo '        <display-name>'${HOST}' Server Network</display-name>' >> 		$CONFIG_FILE_NAME;
echo '        <value/>' >> 														$CONFIG_FILE_NAME;
echo '        <description>'${HOST}' Network Select</description>' >>			$CONFIG_FILE_NAME;
echo '        <value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '            <type>value-list</type>' >> 									$CONFIG_FILE_NAME;
echo '            <entries>' >> 												$CONFIG_FILE_NAME;

IFS=$'|';
HOST_DEVICE=(${OSS_NETWORK[${i}]});
unset IFS;
for LINE in "${HOST_DEVICE[@]}"; do
	echo '            <entry><value>tcp('${LINE}')</value><label>'${LINE}'</label></entry>' >> 				$CONFIG_FILE_NAME;
done

echo '            </entries>' >> 												$CONFIG_FILE_NAME;
echo '            <selection-cardinality>2+</selection-cardinality>' >> 		$CONFIG_FILE_NAME;
echo '        </value-attributes>' >> 											$CONFIG_FILE_NAME;
echo '    </property>' >> 														$CONFIG_FILE_NAME;

i=$((i+1));
done


echo '    <property>' >> 														$CONFIG_FILE_NAME;
echo '        <name>network_client</name>' >> 									$CONFIG_FILE_NAME;
echo '        <display-name>Client Server Network</display-name>' >> 			$CONFIG_FILE_NAME;
echo '        <value>tcp(enp0s8)</value>' >> 									$CONFIG_FILE_NAME;
echo '        <description>Client Network</description>' >>						$CONFIG_FILE_NAME;
echo '    </property>' >> 														$CONFIG_FILE_NAME;
echo '    <property>' >> 														$CONFIG_FILE_NAME;
echo '        <name>mount_client</name>' >> 									$CONFIG_FILE_NAME;
echo '        <display-name>Client Server Mount</display-name>' >> 				$CONFIG_FILE_NAME;
echo '        <value>'${Client[0]}'</value>' >> 								$CONFIG_FILE_NAME;
echo '        <description>Client Mount</description>' >>						$CONFIG_FILE_NAME;
echo '    </property>' >> 														$CONFIG_FILE_NAME;
echo '</configuration>' >> 														$CONFIG_FILE_NAME;







# 기본 세팅파일 저장
echo '<?xml version="1.0"?>' > 													$ADVENCE_CONFIG_FILE_NAME;
echo '<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>' >> 			$ADVENCE_CONFIG_FILE_NAME;
echo '<configuration supports_final="false">' >> 								$ADVENCE_CONFIG_FILE_NAME;
echo '    <property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '        <name>mds_host</name>' >> 										$ADVENCE_CONFIG_FILE_NAME;
echo '        <display-name>MDS Server Host List</display-name>' >> 			$ADVENCE_CONFIG_FILE_NAME;
echo -n '        <value>'${MDS[@]}'</value>' >> 								$ADVENCE_CONFIG_FILE_NAME;

echo '        <description>Client Network</description>' >>						$ADVENCE_CONFIG_FILE_NAME;
echo '    </property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '    <property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '        <name>mdt_fsname</name>' >> 										$ADVENCE_CONFIG_FILE_NAME;
echo '        <value>'${MDS_FSNAME[0]}'</value>' >> 							$ADVENCE_CONFIG_FILE_NAME;

echo '    </property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '    <property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '        <name>mdt_index</name>' >> 										$ADVENCE_CONFIG_FILE_NAME;
echo '        <value>0</value>' >> 												$ADVENCE_CONFIG_FILE_NAME;
echo '    </property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '    <property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '        <name>mdt_mount</name>' >> 										$ADVENCE_CONFIG_FILE_NAME;
echo '        <value>'${MDS_MOUNT[0]}'</value>' >> 								$ADVENCE_CONFIG_FILE_NAME;
echo '    </property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '    <property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '        <name>oss_host</name>' >> 										$ADVENCE_CONFIG_FILE_NAME;
echo -n '        <value>' >> 														$ADVENCE_CONFIG_FILE_NAME;
i=0
for HOST in "${OSS[@]}"; do
echo ${HOST}'|'${OSS_DISKS[$i]} >> 												$ADVENCE_CONFIG_FILE_NAME;
i=$((i+1));
done
echo '        </value>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '    </property>' >> 														$ADVENCE_CONFIG_FILE_NAME;
echo '</configuration>' >> 														$ADVENCE_CONFIG_FILE_NAME;





# theme.json 파일생성
echo '{' >	 																					$THEME_FILE_NAME;
echo '    "name": "default",' >> 																$THEME_FILE_NAME;
echo '    "description": "Default theme for Lustre service",' >> 								$THEME_FILE_NAME;
echo '    "configuration": {' >> 																$THEME_FILE_NAME;
echo '        "layouts": [' >> 																	$THEME_FILE_NAME;
echo '            {' >> 																		$THEME_FILE_NAME;
echo '                "name": "default",' >> 													$THEME_FILE_NAME;
echo '                "tabs": [' >> 															$THEME_FILE_NAME;
# mds 탭 & 레이아웃 설정
echo '                    {' >> 																$THEME_FILE_NAME;
echo '                        "name": "mds-server",' >> 										$THEME_FILE_NAME;
echo '                        "display-name": "MDS Server Setting",' >> 						$THEME_FILE_NAME;
echo '                        "layout": {' >> 													$THEME_FILE_NAME;
echo '                            "tab-columns": "1",' >> 										$THEME_FILE_NAME;
echo '                            "tab-rows": "'${#MDS[@]}'",' >> 								$THEME_FILE_NAME;
echo '                            "sections": [' >> 											$THEME_FILE_NAME;
i=0;
for HOST in "${MDS[@]}"; do

if [[ ("$i" != "0") ]]; then
echo '                                ,' >> 													$THEME_FILE_NAME;
fi

echo '                                {' >> 													$THEME_FILE_NAME;
echo '                                    "name": "section-mds'$((i+1))'-server",' >> 			$THEME_FILE_NAME;
echo '                                    "display-name": "MDS'$((i+1))' Setting",' >> 			$THEME_FILE_NAME;
echo '                                    "row-index": "'$((i))'",' >> 							$THEME_FILE_NAME;
echo '                                    "column-index": "0",' >> 								$THEME_FILE_NAME;
echo '                                    "row-span": "1",' >> 									$THEME_FILE_NAME;
echo '                                    "column-span": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "section-columns": "1",' >> 							$THEME_FILE_NAME;
echo '                                    "section-rows": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "subsections": [' >> 									$THEME_FILE_NAME;
echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-mds-col1-'${HOST}'",' >> 	$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        },' >> 											$THEME_FILE_NAME;
echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-mds-col2-'${HOST}'",' >> 	$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "1",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        },' >> 											$THEME_FILE_NAME;
echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-mds-col3-'${HOST}'",' >> 	$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "2",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        }' >> 											$THEME_FILE_NAME;
echo '                                    ]' >> 												$THEME_FILE_NAME;
echo '                                }' >> 													$THEME_FILE_NAME;
i=$((i+1));
done

echo '                           	]' >> 														$THEME_FILE_NAME;
echo '                        }' >> 															$THEME_FILE_NAME;
echo '                    },' >> 																$THEME_FILE_NAME;



# # oss 탭 & 레이아웃 설정
# echo '                    {' >> 																$THEME_FILE_NAME;
# echo '                        "name": "oss-server",' >> 										$THEME_FILE_NAME;
# echo '                        "display-name": "OSS Server Setting",' >> 						$THEME_FILE_NAME;
# echo '                        "layout": {' >> 													$THEME_FILE_NAME;
# echo '                            "tab-columns": "1",' >> 										$THEME_FILE_NAME;
# echo '                            "tab-rows": "'${#OSS[@]}'",' >> 								$THEME_FILE_NAME;
# echo '                            "sections": [' >> 											$THEME_FILE_NAME;
# i=0;
# for HOST in "${OSS[@]}"; do

# if [[ ("$i" != "0") ]]; then
# echo -e '                                ,' >> 													$THEME_FILE_NAME;
# fi

# echo '                                {' >> 													$THEME_FILE_NAME;
# echo '                                    "name": "section-oss'$((i+1))'-server",' >> 			$THEME_FILE_NAME;
# echo '                                    "display-name": "OSS'$((i+1))' Setting",' >> 			$THEME_FILE_NAME;
# echo '                                    "row-index": "'$((i))'",' >> 							$THEME_FILE_NAME;
# echo '                                    "column-index": "0",' >> 								$THEME_FILE_NAME;
# echo '                                    "row-span": "1",' >> 									$THEME_FILE_NAME;
# echo '                                    "column-span": "1",' >> 								$THEME_FILE_NAME;
# echo '                                    "section-columns": "1",' >> 							$THEME_FILE_NAME;
# echo '                                    "section-rows": "1",' >> 								$THEME_FILE_NAME;
# echo '                                    "subsections": [' >> 									$THEME_FILE_NAME;
# echo '                                        {' >> 											$THEME_FILE_NAME;
# echo '                                            "name": "subsection-oss-col1-'${HOST}'",' >> 	$THEME_FILE_NAME;
# echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
# echo '                                            "column-index": "0",' >> 						$THEME_FILE_NAME;
# echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
# echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
# echo '                                        },' >> 											$THEME_FILE_NAME;
# echo '                                        {' >> 											$THEME_FILE_NAME;
# echo '                                            "name": "subsection-oss-col2-'${HOST}'",' >> 	$THEME_FILE_NAME;
# echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
# echo '                                            "column-index": "1",' >> 						$THEME_FILE_NAME;
# echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
# echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
# echo '                                        },' >> 											$THEME_FILE_NAME;
# echo '                                        {' >> 											$THEME_FILE_NAME;
# echo '                                            "name": "subsection-oss-col3-'${HOST}'",' >> 	$THEME_FILE_NAME;
# echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
# echo '                                            "column-index": "2",' >> 						$THEME_FILE_NAME;
# echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
# echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
# echo '                                        }' >> 											$THEME_FILE_NAME;
# echo '                                    ]' >> 												$THEME_FILE_NAME;
# echo '                                }' >> 													$THEME_FILE_NAME;
# i=$((i+1));
# done

# echo '                           	]' >> 														$THEME_FILE_NAME;
# echo '                        }' >> 															$THEME_FILE_NAME;
# echo '                    },' >> 																$THEME_FILE_NAME;





echo '                    {' >> 																$THEME_FILE_NAME;
echo '                        "name": "oss-server-device",' >> 										$THEME_FILE_NAME;
echo '                        "display-name": "OSS Server Setting (device)",' >> 						$THEME_FILE_NAME;
echo '                        "layout": {' >> 													$THEME_FILE_NAME;
echo '                            "tab-columns": "1",' >> 										$THEME_FILE_NAME;
echo '                            "tab-rows": "'${#OSS[@]}'",' >> 								$THEME_FILE_NAME;
echo '                            "sections": [' >> 											$THEME_FILE_NAME;
i=0;
for HOST in "${OSS[@]}"; do

if [[ ("$i" != "0") ]]; then
echo -e '                                ,' >> 													$THEME_FILE_NAME;
fi

echo '                                {' >> 													$THEME_FILE_NAME;
echo '                                    "name": "section-oss'$((i+1))'-server-device",' >> 			$THEME_FILE_NAME;
echo '                                    "display-name": "OSS'$((i+1))' Device Setting",' >> 			$THEME_FILE_NAME;
echo '                                    "row-index": "'$((i))'",' >> 							$THEME_FILE_NAME;
echo '                                    "column-index": "0",' >> 								$THEME_FILE_NAME;
echo '                                    "row-span": "1",' >> 									$THEME_FILE_NAME;
echo '                                    "column-span": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "section-columns": "1",' >> 							$THEME_FILE_NAME;
echo '                                    "section-rows": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "subsections": [' >> 									$THEME_FILE_NAME;
echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-oss-col1-'${HOST}'",' >> 	$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        }' >> 											$THEME_FILE_NAME;

echo '                                    ]' >> 												$THEME_FILE_NAME;
echo '                                }' >> 													$THEME_FILE_NAME;
i=$((i+1));
done

echo '                           	]' >> 														$THEME_FILE_NAME;
echo '                        }' >> 															$THEME_FILE_NAME;
echo '                    },' >> 																$THEME_FILE_NAME;



# oss 탭 & 레이아웃 설정

echo '                    {' >> 																$THEME_FILE_NAME;
echo '                        "name": "oss-server-devicesize",' >> 										$THEME_FILE_NAME;
echo '                        "display-name": "OSS Server Setting (device size)",' >> 						$THEME_FILE_NAME;
echo '                        "layout": {' >> 													$THEME_FILE_NAME;
echo '                            "tab-columns": "1",' >> 										$THEME_FILE_NAME;
echo '                            "tab-rows": "'${#OSS[@]}'",' >> 								$THEME_FILE_NAME;
echo '                            "sections": [' >> 											$THEME_FILE_NAME;
i=0;
for HOST in "${OSS[@]}"; do

if [[ ("$i" != "0") ]]; then
echo -e '                                ,' >> 													$THEME_FILE_NAME;
fi

echo '                                {' >> 													$THEME_FILE_NAME;
echo '                                    "name": "section-oss'$((i+1))'-server-devicesize",' >> 			$THEME_FILE_NAME;
echo '                                    "display-name": "OSS'$((i+1))' Device Size Setting",' >> 			$THEME_FILE_NAME;
echo '                                    "row-index": "'$((i))'",' >> 							$THEME_FILE_NAME;
echo '                                    "column-index": "0",' >> 								$THEME_FILE_NAME;
echo '                                    "row-span": "1",' >> 									$THEME_FILE_NAME;
echo '                                    "column-span": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "section-columns": "1",' >> 							$THEME_FILE_NAME;
echo '                                    "section-rows": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "subsections": [' >> 									$THEME_FILE_NAME;

echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-oss-col2-'${HOST}'",' >> 	$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        }' >> 											$THEME_FILE_NAME;

echo '                                    ]' >> 												$THEME_FILE_NAME;
echo '                                }' >> 													$THEME_FILE_NAME;
i=$((i+1));
done

echo '                           	]' >> 														$THEME_FILE_NAME;
echo '                        }' >> 															$THEME_FILE_NAME;
echo '                    },' >> 																$THEME_FILE_NAME;




# oss 탭 & 레이아웃 설정

echo '                    {' >> 																$THEME_FILE_NAME;
echo '                        "name": "oss-server-network",' >> 										$THEME_FILE_NAME;
echo '                        "display-name": "OSS Server Setting (network)",' >> 						$THEME_FILE_NAME;
echo '                        "layout": {' >> 													$THEME_FILE_NAME;
echo '                            "tab-columns": "1",' >> 										$THEME_FILE_NAME;
echo '                            "tab-rows": "'${#OSS[@]}'",' >> 								$THEME_FILE_NAME;
echo '                            "sections": [' >> 											$THEME_FILE_NAME;
i=0;
for HOST in "${OSS[@]}"; do

if [[ ("$i" != "0") ]]; then
echo -e '                                ,' >> 													$THEME_FILE_NAME;
fi

echo '                                {' >> 													$THEME_FILE_NAME;
echo '                                    "name": "section-oss'$((i+1))'-server-network",' >> 			$THEME_FILE_NAME;
echo '                                    "display-name": "OSS'$((i+1))' Network Setting",' >> 			$THEME_FILE_NAME;
echo '                                    "row-index": "'$((i))'",' >> 							$THEME_FILE_NAME;
echo '                                    "column-index": "0",' >> 								$THEME_FILE_NAME;
echo '                                    "row-span": "1",' >> 									$THEME_FILE_NAME;
echo '                                    "column-span": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "section-columns": "1",' >> 							$THEME_FILE_NAME;
echo '                                    "section-rows": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "subsections": [' >> 									$THEME_FILE_NAME;

echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-oss-col3-'${HOST}'",' >> 	$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        }' >> 											$THEME_FILE_NAME;
echo '                                    ]' >> 												$THEME_FILE_NAME;
echo '                                }' >> 													$THEME_FILE_NAME;
i=$((i+1));
done

echo '                           	]' >> 														$THEME_FILE_NAME;
echo '                        }' >> 															$THEME_FILE_NAME;
echo '                    },' >> 																$THEME_FILE_NAME;



# client 탭 & 레이아웃 설정
echo '                    {' >> 																$THEME_FILE_NAME;
echo '                        "name": "client-server",' >> 										$THEME_FILE_NAME;
echo '                        "display-name": "Client Server Setting",' >> 						$THEME_FILE_NAME;
echo '                        "layout": {' >> 													$THEME_FILE_NAME;
echo '                            "tab-columns": "1",' >> 										$THEME_FILE_NAME;
echo '                            "tab-rows": "1",' >> 											$THEME_FILE_NAME;
echo '                            "sections": [' >> 											$THEME_FILE_NAME;
echo '                                {' >> 													$THEME_FILE_NAME;
echo '                                    "name": "section-client-server",' >> 					$THEME_FILE_NAME;
echo '                                    "display-name": "Client Setting",' >> 				$THEME_FILE_NAME;
echo '                                    "row-index": "0",' >> 								$THEME_FILE_NAME;
echo '                                    "column-index": "0",' >> 								$THEME_FILE_NAME;
echo '                                    "row-span": "1",' >> 									$THEME_FILE_NAME;
echo '                                    "column-span": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "section-columns": "1",' >> 							$THEME_FILE_NAME;
echo '                                    "section-rows": "1",' >> 								$THEME_FILE_NAME;
echo '                                    "subsections": [' >> 									$THEME_FILE_NAME;
echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-client-col1",' >> 		$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        },' >> 											$THEME_FILE_NAME;
echo '                                        {' >> 											$THEME_FILE_NAME;
echo '                                            "name": "subsection-client-col2",' >> 		$THEME_FILE_NAME;
echo '                                            "row-index": "0",' >> 						$THEME_FILE_NAME;
echo '                                            "column-index": "1",' >> 						$THEME_FILE_NAME;
echo '                                            "row-span": "1",' >> 							$THEME_FILE_NAME;
echo '                                            "column-span": "1"' >> 						$THEME_FILE_NAME;
echo '                                        }' >> 											$THEME_FILE_NAME;
echo '                                    ]' >> 												$THEME_FILE_NAME;
echo '                                }' >> 													$THEME_FILE_NAME;
echo '                            ]' >> 														$THEME_FILE_NAME;
echo '                        }' >> 															$THEME_FILE_NAME;

echo '                    }' >> 																$THEME_FILE_NAME;

echo '                ]' >> 																	$THEME_FILE_NAME;
echo '            }' >> 																		$THEME_FILE_NAME;
echo '        ],' >> 																			$THEME_FILE_NAME;

echo '        "placement": {' >> $THEME_FILE_NAME;
echo '            "configuration-layout": "default",' >> 										$THEME_FILE_NAME;
echo '            "configs": [' >> 																$THEME_FILE_NAME;


# mds 드라이브 선택부분
i=0;
for HOST in "${MDS[@]}"; do
# 콤마출력
# if [[ ("$i" != "0") ]]; then
# 	echo ',' >> 																				$THEME_FILE_NAME;
# fi
	echo '            {' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/device_mds-'${HOST}'",' >> 		$THEME_FILE_NAME;
	echo '                    "subsection-name": "subsection-mds-col1-'${HOST}'"' >>	$THEME_FILE_NAME;
	echo '            },' >> 																	$THEME_FILE_NAME;

	# echo '            {' >> 																	$THEME_FILE_NAME;
	# echo '                    "config": "lustre-default-config/device_mds_size-'${HOST}'",' >> 		$THEME_FILE_NAME;
	# echo '                    "subsection-name": "subsection-mds-col2-'${HOST}'"' >>	$THEME_FILE_NAME;
	# echo '            },' >> 																	$THEME_FILE_NAME;

	j=0;
	IFS=$'|';
	HOST_DEVICE=(${MDS_DF[${i}]});
	unset IFS;
	for LINE in "${HOST_DEVICE[@]}"; do
	echo '            {' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/device_mds_size-'${HOST}'-'${LINE////}'",' >> 		$THEME_FILE_NAME;
	echo '                    "subsection-name": "subsection-mds-col2-'${HOST}'"' >>	$THEME_FILE_NAME;
	echo '            },' >> 																	$THEME_FILE_NAME;
	done

	echo '            {' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/network_mds-'${HOST}'",' >> 		$THEME_FILE_NAME;
	echo '                    "subsection-name": "subsection-mds-col3-'${HOST}'"' >>	$THEME_FILE_NAME;
	echo '            },' >> 																	$THEME_FILE_NAME;
i=$((i+1));
done



# oss 드라이브 선택부분
i=0;
for HOST in "${OSS[@]}"; do
	j=0; LIMIT=${OSS_DISKS[$i]};

# # 콤마출력
# if [[ ("$i" != "0") ]]; then
# 	echo ',' >> 																				$THEME_FILE_NAME;
# fi
	echo '            {' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/network_oss-'${HOST}'",' >> 		$THEME_FILE_NAME;
	echo '                    "subsection-name": "subsection-oss-col3-'${HOST}'"' >>	$THEME_FILE_NAME;
	echo '            },' >> 																	$THEME_FILE_NAME;	

	# while (( j < LIMIT )); do
	# 	echo '            ,{' >> 																	$THEME_FILE_NAME;
	# 	echo '                    "config": "lustre-default-config/device_oss'$((j+1))'-'${HOST}'",' >> 		$THEME_FILE_NAME;
	# 	echo '                    "subsection-name": "subsection-oss-col1-'${HOST}'"' >>	$THEME_FILE_NAME;
	# 	echo '            },' >> 																	$THEME_FILE_NAME;
	# 	echo '            {' >> 																	$THEME_FILE_NAME;
	# 	echo '                    "config": "lustre-default-config/device_oss_size'$((j+1))'-'${HOST}'",' >> 		$THEME_FILE_NAME;
	# 	echo '                    "subsection-name": "subsection-oss-col2-'${HOST}'"' >>	$THEME_FILE_NAME;
	# 	echo '            }' >> 																	$THEME_FILE_NAME;
	# 	j=$((j+1));
	# done

	while (( j < LIMIT )); do
		echo '            {' >> 																	$THEME_FILE_NAME;
		echo '                    "config": "lustre-default-config/device_oss'$((j+1))'-'${HOST}'",' >> 		$THEME_FILE_NAME;
		echo '                    "subsection-name": "subsection-oss-col1-'${HOST}'"' >>	$THEME_FILE_NAME;
		echo '            },' >> 																	$THEME_FILE_NAME;
		
		j=$((j+1));
	done

	k=0;
	IFS=$'|';
	HOST_DEVICE=(${OSS_DF[${i}]});
	unset IFS;
	for LINE in "${HOST_DEVICE[@]}"; do
		echo '            {' >> 																	$THEME_FILE_NAME;
		echo '                    "config": "lustre-default-config/device_oss_size-'${HOST}'-'${LINE////}'",' >> 		$THEME_FILE_NAME;
		echo '                    "subsection-name": "subsection-oss-col2-'${HOST}'"' >>	$THEME_FILE_NAME;
		echo '            },' >> 																	$THEME_FILE_NAME;
		k=$((k+1));
	done

i=$((i+1));
done

	echo '            {' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/network_client",' >> 		$THEME_FILE_NAME;
	echo '                    "subsection-name": "subsection-client-col1"' >>	$THEME_FILE_NAME;
	echo '            }' >> 																	$THEME_FILE_NAME;
	echo '            ,{' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/mount_client",' >> 		$THEME_FILE_NAME;
	echo '                    "subsection-name": "subsection-client-col2"' >>	$THEME_FILE_NAME;
	echo '            }' >> 																	$THEME_FILE_NAME;


echo '				]' >> $THEME_FILE_NAME;
echo '        },' >> $THEME_FILE_NAME;


echo '        "widgets": [' >> $THEME_FILE_NAME;

# mds 드라이브 선택부분
i=0;
for HOST in "${MDS[@]}"; do
# 콤마출력
if [[ ("$i" != "0") ]]; then
	echo ',' >> 																				$THEME_FILE_NAME;
fi
	echo '            {' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/device_mds-'${HOST}'",' >> 		$THEME_FILE_NAME;
	echo '                	  "widget": {' >> $THEME_FILE_NAME;
	echo '                	      "type": "combo"' >> $THEME_FILE_NAME;
	echo '                	  }' >> $THEME_FILE_NAME;
	echo '            },' >> 																	$THEME_FILE_NAME;

	# echo '            {' >> 																		$THEME_FILE_NAME;
	# echo '                    "config": "lustre-default-config/device_mds_size-'${HOST}'",' >> 		$THEME_FILE_NAME;
	# echo '                	  "widget": {' >> 														$THEME_FILE_NAME;
	# echo '                	      "type": "slider",' >> 											$THEME_FILE_NAME;
	# echo '               		   "units": [' >> 													$THEME_FILE_NAME;
	# echo '               		      {' >> 														$THEME_FILE_NAME;
	# echo '               		        "unit-name":"GB"' >> 										$THEME_FILE_NAME;
	# echo '               		      }' >> 														$THEME_FILE_NAME;
	# echo '               		   ]' >> 															$THEME_FILE_NAME;
	# echo '                	  }' >> 																$THEME_FILE_NAME;
	# echo '            },' >> 																	$THEME_FILE_NAME;
	j=0;
	IFS=$'|';
	HOST_DEVICE_SIZE=(${MDS_DF[${i}]});
	unset IFS;
	for LINE in "${HOST_DEVICE_SIZE[@]}"; do
	echo '            {' >> 																		$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/device_mds_size-'${HOST}'-'${LINE////}'",' >> 		$THEME_FILE_NAME;
	echo '                	  "widget": {' >> 														$THEME_FILE_NAME;
	echo '                	      "type": "slider",' >> 											$THEME_FILE_NAME;
	echo '               		   "units": [' >> 													$THEME_FILE_NAME;
	echo '               		      {' >> 														$THEME_FILE_NAME;
	echo '               		        "unit-name":"GB"' >> 										$THEME_FILE_NAME;
	echo '               		      }' >> 														$THEME_FILE_NAME;
	echo '               		   ]' >> 															$THEME_FILE_NAME;
	echo '                	  }' >> 																$THEME_FILE_NAME;
	echo '            },' >> 																	$THEME_FILE_NAME;

	k=$((k+1));
	done

	echo '            {' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/network_mds-'${HOST}'",' >> 		$THEME_FILE_NAME;
	echo '                	  "widget": {' >> $THEME_FILE_NAME;
	echo '                	      "type": "combo"' >> $THEME_FILE_NAME;
	echo '                	  }' >> $THEME_FILE_NAME;
	echo '            }' >> 																	$THEME_FILE_NAME;
i=$((i+1));
done

# oss 드라이브 선택부분
i=0;
for HOST in "${OSS[@]}"; do
	j=0; LIMIT=${OSS_DISKS[$i]};


	echo '            ,{' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/network_oss-'${HOST}'",' >> 		$THEME_FILE_NAME;
	echo '                	  "widget": {' >> $THEME_FILE_NAME;
	echo '                	      "type": "combo"' >> $THEME_FILE_NAME;
	echo '                	  }' >> $THEME_FILE_NAME;
	echo '            }' >> 																	$THEME_FILE_NAME;

	# while (( j < LIMIT )); do
	# 	# if [[ ("$j" != "0") ]]; then
	# 	# 	echo ',' >> 																				$THEME_FILE_NAME;
	# 	# fi
	# 	echo '            ,{' >> 																	$THEME_FILE_NAME;
	# 	echo '                    "config": "lustre-default-config/device_oss'$((j+1))'-'${HOST}'",' >> 		$THEME_FILE_NAME;
	# 	echo '                	  "widget": {' >> $THEME_FILE_NAME;
	# 	echo '                	      "type": "combo"' >> $THEME_FILE_NAME;
	# 	echo '                	  }' >> $THEME_FILE_NAME;
	# 	echo '            },' >> 																	$THEME_FILE_NAME;
	# 	echo '            {' >> 																	$THEME_FILE_NAME;
	# 	echo '                    "config": "lustre-default-config/device_oss_size'$((j+1))'-'${HOST}'",' >> 		$THEME_FILE_NAME;
	# 	echo '                	  "widget": {' >> $THEME_FILE_NAME;
	# 	echo '                	      "type": "slider",' >> 											$THEME_FILE_NAME;
	# 	echo '               		   "units": [' >> 													$THEME_FILE_NAME;
	# 	echo '               		      {' >> 														$THEME_FILE_NAME;
	# 	echo '               		        "unit-name":"GB"' >> 										$THEME_FILE_NAME;
	# 	echo '               		      }' >> 														$THEME_FILE_NAME;
	# 	echo '               		   ]' >> 															$THEME_FILE_NAME;
	# 	echo '                	  }' >> $THEME_FILE_NAME;
	# 	echo '            }' >> 																	$THEME_FILE_NAME;
	# j=$((j+1));
	# done

	while (( j < LIMIT )); do
		# if [[ ("$j" != "0") ]]; then
		# 	echo ',' >> 																				$THEME_FILE_NAME;
		# fi
		echo '            ,{' >> 																	$THEME_FILE_NAME;
		echo '                    "config": "lustre-default-config/device_oss'$((j+1))'-'${HOST}'",' >> 		$THEME_FILE_NAME;
		echo '                	  "widget": {' >> $THEME_FILE_NAME;
		echo '                	      "type": "combo"' >> $THEME_FILE_NAME;
		echo '                	  }' >> $THEME_FILE_NAME;
		echo '            }' >> 																	$THEME_FILE_NAME;
	j=$((j+1));
	done

	k=0;
	IFS=$'|';
	HOST_DEVICE_SIZE=(${OSS_DF[${k}]});
	unset IFS;
	for LINE in "${HOST_DEVICE_SIZE[@]}"; do
		echo '            ,{' >> 																	$THEME_FILE_NAME;
		echo '                    "config": "lustre-default-config/device_oss_size-'${HOST}'-'${LINE////}'",' >> 		$THEME_FILE_NAME;
		echo '                	  "widget": {' >> $THEME_FILE_NAME;
		echo '                	      "type": "slider",' >> 											$THEME_FILE_NAME;
		echo '               		   "units": [' >> 													$THEME_FILE_NAME;
		echo '               		      {' >> 														$THEME_FILE_NAME;
		echo '               		        "unit-name":"GB"' >> 										$THEME_FILE_NAME;
		echo '               		      }' >> 														$THEME_FILE_NAME;
		echo '               		   ]' >> 															$THEME_FILE_NAME;
		echo '                	  }' >> $THEME_FILE_NAME;
		echo '            }' >> 																	$THEME_FILE_NAME;
		k=$((k+1));
	done

# # 콤마출력
# if [[ ("$i" != "0") ]]; then
# 	echo ',' >> 																				$THEME_FILE_NAME;
# fi

i=$((i+1));
done

	echo '            ,{' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/network_client",' >> 		$THEME_FILE_NAME;
	echo '                	  "widget": {' >> $THEME_FILE_NAME;
	echo '                	      "type":"text-field"' >> $THEME_FILE_NAME;
	echo '                	  }' >> $THEME_FILE_NAME;
	echo '            }' >> 																	$THEME_FILE_NAME;
	echo '            ,{' >> 																	$THEME_FILE_NAME;
	echo '                    "config": "lustre-default-config/mount_client",' >> 		$THEME_FILE_NAME;
	echo '                	  "widget": {' >> $THEME_FILE_NAME;
	echo '                	      "type":"text-field"' >> $THEME_FILE_NAME;
	echo '                	  }' >> $THEME_FILE_NAME;
	echo '            }' >> 	$THEME_FILE_NAME;

echo '        ]' >> $THEME_FILE_NAME;


echo '    }' >> 																				$THEME_FILE_NAME;
echo '}' >> 																					$THEME_FILE_NAME;


cp -b $CONFIG_FILE_PATH'/'$CONFIG_FILE $CONFIG_FILE_NAME'.bak'
cp -b $THEME_FILE_PATH'/'$THEME_FILE $THEME_FILE_NAME'.bak'

cp -b $ADVENCE_CONFIG_FILE_PATH'/'$ADVENCE_CONFIG_FILE $ADVENCE_CONFIG_FILE_NAME'.bak'


cp -f $ADVENCE_CONFIG_FILE_NAME $ADVENCE_CONFIG_FILE_PATH

cp -f $CONFIG_FILE_NAME $CONFIG_FILE_PATH
cp -f $THEME_FILE_NAME $THEME_FILE_PATH


# ambari-server restart










