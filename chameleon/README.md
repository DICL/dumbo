# Chameleon

## Introduction

  Chameleon is a project to develop a HPC based big-data platform operation management system. 

  High performance computing (HPC) community is increasingly demanding big data processing beyond traditional simulation-based computation. Hadoop ecosystem has a roadmap that includes HPC support including GPU, FPGA. With HPC and Big-data converging into one huge ecosystem, we launched the Chameleon project to develop a HPC based big-data platform operation management system. 

  Chameleon aims to integrate operations management of the [Hadoop](https://hadoop.apache.org/) platform for Big-data and [Lustre](http://lustre.org/) file system for HPC, helping HPC and big-data two communities to move into one huge ecosystem. Specifically, it contributes to expanding open-source [Ambari](https://ambari.apache.org/) to HPC communities which needs Big-data processing on top of traditional HPC infrastructure.

## Main Functionalities

  Chameleon was developed based on Apache [Ambari](https://ambari.apache.org/), which is well-known [Hadoop](https://hadoop.apache.org/) management system and extended to support [Lustre](http://lustre.org/) filesystem management, which is widely used in HPC community for massive storage and HPC resource monitoring including GPU and Infiniband. It has the following functional features:

### Integration of the the Ambari stack and Lustre stack
* Define Lustre stack with components like Hadoop stack in Ambari

| Stack  | Component |
| ------ | ----------|
| HDFS | Master(NameNode), Slave(DataNode), Client(HDFS Client) |
| LustreKernelUpdater | Master(LustreFSKernelMaster), Client(LustreFSKernelClient) |
| LustreManager | Master(HadoopLustreAdapterMgmtService, LustreMDSMgmtService) Slave(LustreOSSMgmtService), Client(LustreClient) |
| AccountManager | Master(AccountManagerMaster), Client(AccountManagerClient) |


### LustreKernelUpdater Service
* Support LustreFS Kernel installation on MDS and OSS server nodes

### LustreManager Service
* Enable provisioning of LustreFS with failover using Ambari stack

### AccountManager Service
* Integrate account management between Big Data Platform and LustreFS via [LDAP](https://en.wikipedia.org/wiki/Lightweight_Directory_Access_Protocol)

<img src="https://github.com/DICL/dumbo/blob/master/chameleon/images/AccountManagerView_img.png" width="70%" height="70%"></img>


### Hadoop on LustreFS
* Create working directory
* Switch between HDFS and LustreFS for Hadoop applications


### Advanced LustreFS management
* Provide configurations and mounting for Lustre file systems
* Manage OST, MDT, and Lnet
* Backup and restore of OST


### Advanced YARN application monitoring 
* Create user-defined metrics using linux performance tools for [YARN](https://hadoop.apache.org/docs/current/hadoop-yarn/hadoop-yarn-site/YARN.html) app monitoring

<img src="https://github.com/DICL/dumbo/blob/master/chameleon/images/MetricRegistryView_img.PNG" width="70%" height="70%"></img>

* Support time-series data analysis per applications, metrics, or nodes 


<img src="https://github.com/DICL/dumbo/blob/master/chameleon/images/TimeSereisAnalysis_img.png" width="70%" height="70%"></img>


### Dashboard for dynamic user interface
* Build dynamic dashboard for Hadoop and HPC which streamlines HPC based Big-data platform operation and management


## Demo Video
[![201905 demo](https://img.youtube.com/vi/MQtMwB8k_cE/0.jpg)](https://youtu.be/MQtMwB8k_cE)
