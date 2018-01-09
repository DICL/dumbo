# This file contains environment variables that are passed to mesos-agent.
# To get a description of all options run mesos-agent --help; any option
# supported as a command-line option is also supported as an environment
# variable.

# You must at least set MESOS_master.

# The mesos master URL to contact. Should be host:port for
# non-ZooKeeper based masters, otherwise a zk:// or file:// URL.
export MESOS_master=unknown-machine:5050

# Other options you're likely to want to set:
# export MESOS_log_dir=/var/log/mesos
# export MESOS_work_dir=/var/run/mesos
# export MESOS_isolation=cgroups

export MESOS_hadoop_home=/home/jojaeeon/bigdata/hadoop/hadoop-mapreduce1-project/build/hadoop-2.6.0-mr1-cdh5.11.1
export MESOS_ip=$(getent hosts $(hostname) | cut -d' ' -f1)
export MESOS_master=10.0.5.100:5050
export MESOS_work_dir=/var/lib/mesos
export MESOS_log_dir=/home/jojaeeon/bigdata/run/log/$(hostname)
export MESOS_systemd_enable_support=false
