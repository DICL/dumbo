import sys
import os
import subprocess
import ConfigParser
import time
import random
import commands
import socket
import json
from subtools import SubTools
from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig

try:
    import requests
except ImportError:
    file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(file_path + '/modules')
    import requests


class HadoopLustreAdapterMgmtService(Script):
    def __init__(self):
        self.sbtls = SubTools()

    def install(self, env):
        import params;

        print 'Install HadoopLustreAdapterMgmtService.'

        # Load the all configuration files
        # config = Script.get_config()
        # Bind to a local variable
        # HadoopLustreAdapterMgmtService_user = config['configurations']['my-config-env']['HadoopLustreAdapterMgmtService_user']


        #mds_host = params.mds_host;
        # Install packages
        self.install_packages(env)

        ## init process

        Execute(format("yum install wget libselinux-devel nfs-utils-lib-devel -y"))
        Execute(format("yum groupinstall development tools -y"))
        # Execute(format("wget -P /tmp http://scteam.ksc.re.kr/~jhkwak/lustre/client/lustre-client-2.7.0-2.6.32_504.8.1.el6.x86_64.src.rpm"))
        # Execute(format("ssh root@localhost -T \"nohup rpmbuild --rebuild --define 'lustre_name lustre-client' --define 'configure_args --disable-server --enable-client' --without servers /tmp/lustre-client-2.7.0-2.6.32_504.8.1.el6.x86_64.src.rpm > /tmp/buildhistory\""))
        # Execute(format("yum localinstall /root/rpmbuild/RPMS/x86_64/lustre-iokit-* -y"))
        # Execute(format("yum localinstall /root/rpmbuild/RPMS/x86_64/lustre-client-* -y"))

        Execute(format("wget -P /tmp https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/client/RPMS/x86_64/kmod-lustre-client-2.10.0-1.el7.x86_64.rpm"))
        Execute(format("wget  -P /tmp https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/client/RPMS/x86_64/lustre-client-2.10.0-1.el7.x86_64.rpm"))
        Execute(format("wget  -P /tmp https://downloads.hpdd.intel.com/public/lustre/lustre-2.10.0/el7/client/RPMS/x86_64/lustre-iokit-2.10.0-1.el7.x86_64.rpm"))

	#shkim 20181016 : kmod-lustre-iokit => kmod-lustre-*
        Execute(format("yum localinstall /tmp/kmod-lustre-* -y"))
        Execute(format("yum localinstall /tmp/lustre-iokit-* -y"))
        Execute(format("yum localinstall /tmp/lustre-client-* -y"))

        # Execute(format("wget http://repos.fedorapeople.org/repos/dchen/apache-maven/epel-apache-maven.repo -O /etc/yum.repos.d/epel-apache-maven.repo"))
        # Execute(format("sed -i s/\$releasever/6/g /etc/yum.repos.d/epel-apache-maven.repo"))
        # Execute(format("yum install -y apache-maven git"))
        # Execute(format("git clone https://github.com/Seagate/lustrefs"))
        # Execute(format("mvn -f lustrefs/ package"))

	# shkim 20181016 : check exists
        directory_path='/mnt/lustre/hadoop'
        if not os.path.exists(directory_path):
            os.makedirs("/mnt/lustre/hadoop")
        # os.system("ssh root@localhost -t \"mkdir -p /mnt/lustre/hadoop\"")

        # Execute(format("/bin/mount -t lustre 192.168.1.2116@tcp:/mylustre  /mnt/lustre"))

        # Execute(format("mkdir /lustre"))

        #os.makedirs("/mnt/lustre/hadoop")
	# shkim 20181016 : -t => -T
        #os.system("ssh root@localhost -T \"mkdir -p /mnt/lustre/hadoop\"")


        # Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-hdfs-namenode/lib/"))
        # Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-yarn-nodemanager/lib/"))
        # Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-mapreduce-client/lib/"))
        # Execute(format("yes | cp  lustrefs/target/lustrefs-hadoop-0.9.1.jar /usr/hdp/current/hadoop-mapreduce-historyserver/lib/"))

        hosts = self.getHostsInfo();
        file_path = os.path.dirname(os.path.realpath(__file__));

        for hostname in hosts:
            #self.subprocess_open('echo "'+hostname+'" >> /home/daemon/dm-log.info')
            self.sbtls.sp_open("scp "+file_path+"/modules/lustrefs-hadoop-0.9.1.jar root@"+hostname+":/usr/hdp/current/hadoop-hdfs-namenode/lib/")
            self.sbtls.sp_open("scp "+file_path+"/modules/lustrefs-hadoop-0.9.1.jar root@"+hostname+":/usr/hdp/current/hadoop-yarn-nodemanager/lib/")
            self.sbtls.sp_open("scp "+file_path+"/modules/lustrefs-hadoop-0.9.1.jar root@"+hostname+":/usr/hdp/current/hadoop-mapreduce-client/lib/")
            self.sbtls.sp_open("scp "+file_path+"/modules/lustrefs-hadoop-0.9.1.jar root@"+hostname+":/usr/hdp/current/hadoop-mapreduce-historyserver/lib/")


        #Execute(format('cp '+file_path+'/modules/Ambari_View/Lustre-View-2.0.0.0-SNAPSHOT.war /var/lib/ambari-server/resources/views/ '));

        print 'Installation complete.'

    def stop(self, env):
        print 'Stop HadoopLustreAdapterMgmtService'
        # Stop your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially remove the PID.
        # Execute( "rm -f /tmp/HadoopLustreAdapterMgmtService.pid" )
        file_path = os.path.dirname(os.path.realpath(__file__))
        self.sbtls.sp_open('python ' + file_path + '/daemon-lustre.py stop')

    def start(self, env):
        print 'Start HadoopLustreAdapterMgmtService'
        # Reconfigure all files
        # Start your service

        #Since we have not installed a real service, there is no PID file created by it.
        #Therefore we are going to artificially create the PID.
        # Execute( "touch /tmp/HadoopLustreAdapterMgmtService.pid" )
        file_path = os.path.dirname(os.path.realpath(__file__));
        self.sbtls.sp_open('python ' + file_path + '/daemon-lustre.py start')

    # def status(self, env):
    #     print 'Status of HadoopLustreAdapterMgmtService'
    #     HadoopLustreAdapterMgmtService_pid_file = "/tmp/HadoopLustreAdapterMgmtService.pid"
    #     #check_process_status(dummy_master_pid_file)
    #     Execute( format("cat {HadoopLustreAdapterMgmtService_pid_file}") )
    #     pass

    def switchtohdfs(self, env):
        self.sbtls.excuteDaemon('tohdfs', 5679)
        print("switchtohdfs!!")

    def switchtolustrefs(self, env):
        print 'init'
        self.sbtls.excuteDaemon('tolustrefs', 5679)
        print 'init-done'
        print("switchtolustrefs!!")

    def getHostsInfo(self):
        global local_hostname
        local_hostname = socket.gethostname()
        print "Local Hostname : " + local_hostname

        # ambari server hostname
        config = ConfigParser.ConfigParser()
        config.read("/etc/ambari-agent/conf/ambari-agent.ini")
        global ambari_url
        ambari_url = config.get('server', 'hostname')
        print "Ambari server Hostname : " + ambari_url

        # cluster_name
        headers = {
            'X-Requested-By': 'ambari',
        }
        r = requests.get(
            'http://' + ambari_url + ':8080/api/v1/clusters',
            headers=headers,
            auth=('admin', 'admin'))
        j = json.loads(r.text)
        items = j["items"][0]["Clusters"]
        print "Cluster Name : " + items["cluster_name"]

        # LustreClient HostNames
        # curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/bigcluster/services/LUSTRE/components/LUSTRE_CLIENT


        r = requests.get(
            'http://' + ambari_url + ':8080/api/v1/clusters/' +
            items["cluster_name"] +
            '/services/LUSTREMANAGER/components/LustreClient',
            headers=headers,
            auth=('admin', 'admin'))
        j = json.loads(r.text)

        result = []
        for component in j["host_components"]:
            result.append(component["HostRoles"]["host_name"])

        return result

    def mount(self, env):
        self.sbtls.excuteDaemon('mount', 5679)
        print("mount!!")

    def umount(self, env):
        self.sbtls.excuteDaemon('umount', 5679)
        print("umount!!")

    def createworkingdir(self, env):
        #self.sbtls.excuteDaemon('createworkingdir', 5679)
        self.sbtls.excuteDaemon('create_workingdir', 5679)
        print("create!!")

    def createworkingdir_bak(self, env):
        self.sbtls.excuteDaemon('createworkingdir', 5679)
        print("create!!")
    
    def removeworkingdir(self, env):
        self.sbtls.excuteDaemon('remove_workingdir', 5679)
        print("remove!!")

    def status(self, env):
        file_path = os.path.dirname(os.path.realpath(__file__))
        check = self.sbtls.sp_open('python '+file_path+'/daemon-lustre.py status')
        print check
        if 'not' in str(check):
            raise ComponentIsNotRunning
        pass


if __name__ == "__main__":
    HadoopLustreAdapterMgmtService().execute()
