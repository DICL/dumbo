# Introduction
The goal of lustre-integrated BigBWA is to complement BigBWA(https://github.com/citiususc/BigBWA) to be executed on lustre file system environment to enhance performance and compatibility. It works with a lustre-hadoop adapter released by Seagate Inc.(https://github.com/Seagate/lustrefs)

# Requirements
- BigBWA(https://github.com/citiususc/BigBWA)
- Apache Hadoop 2.7.3 or later
- Mountable Lustre file system
- Lustre-Hadoop Adapter(integrated in this package, from https://github.com/Seagate/lustrefs) and maven to build it
- MPI

# Install
1. Mount lustre file system and create directory \<lustre mount point\>/hadoop/user/\<username\> and grant permission to user who will run Hadoop
2. Check if MPI is installed by run "mpicc --version" and "mpirun --version"
3. If you use Hadoop 2.7.2, apply patch(MAPRED-6636.patch,https://issues.apache.org/jira/browse/HADOOP-6636) for handling large file(greater than 2GB), build and install
4. Build lustre-hadoop adapter (located in ./lustrefs/) with maven
5. Configure hadoop, 
  in <hadoop home>/etc/hadoop/core-site.xml add

         <property>
             <name>fs.lustrefs.shared_tmp.dir</name>
             <value>${fs.lustrefs.mount}/user/${user.name}/shared-tmp</value>
         </property>

6. Download lustre-integrated BigBWA, copy files in /src to BigBWA/src, and apply patches(Makefile, Makefile.common, src/BigBWA.java) on BigBWA directory
7. Build BigBWA with "make"

# Run
1. Locate input files (pair of .fastq)  under \<lustre mount\>/hadoop/user/\<username\>/
2. Set variable "LUSTRE_ADAPTER" for the location of lustrefs-hadoop jar file and "RG" for the location of reference genome .fa file in build/run.sh
3. On build directory, execute run.sh

        USAGE : run.sh <# of partitions> <# of threads per mapper> <input_1> <input_2> <outputdir>
           * input output file location is relative path from HDFS user home directory

4. An output file will be created on \<lustre mount\>/hadoop/user/\<username\>/\<outputdir\>/merged.sam
