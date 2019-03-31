clear && printf '\033[3J'

yarn jar MOHA.jar org.kisti.moha.MOHA_Client -manager_memory 12000 -executor_memory 6024 -num_executors 6 -jar MOHA.jar -JDL Autodock_Vina.jdl -conf conf/MOHA_ActiveMQ.conf

