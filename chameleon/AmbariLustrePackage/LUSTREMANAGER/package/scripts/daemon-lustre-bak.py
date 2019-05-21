# -*- coding: utf-8 -*-
import commands
import sys, os, subprocess, ConfigParser, time, random, commands, socket, json, SocketServer
from socket import *
from select import *
from daemon import Daemon
from subtools import SubTools
from switch_config import SwitchConfig
from resource_management import *
from resource_management.core.resources.system import File, Execute, Directory
from resource_management.libraries.functions.format import format
from resource_management.core.resources.service import Service
from resource_management.core.exceptions import ComponentIsNotRunning
from resource_management.core import shell
from resource_management.libraries.script.script import Script
from ambari_agent import AmbariConfig

try:
    import requests;
except ImportError:
    file_path = os.path.dirname(os.path.realpath(__file__));
    sys.path.append(file_path+'/modules');
    import requests;


class MyDaemon(Daemon):
  def run(self):
    while True:
      # Do any task here
      time.sleep(1)
      # socket 번호 수정
      # 5678 -> 5679
      HOST, PORT = "localhost", 5679
      server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
      server.serve_forever()

class MyTCPHandler(SocketServer.BaseRequestHandler):
  def handle(self):

    self.data = self.request.recv(1024).strip()
    lustreSomething = LustreSomething()        
    if(str(self.data)=='tolustrefs'):
      lustreSomething.switchToLustreFS()
    elif(str(self.data)=="tohdfs"):
      lustreSomething.switchToHdfs() #예정.      
    elif(str(self.data)=='sample02'):
      lustreSomething.subprocess_open('ls -al /home/daemon > /home/daemon/ls.txt')
    elif(str(self.data)=='sample-time'):
      lustreSomething.do_something()
    elif(str(self.data)=='mount'):
      lustreSomething.mount()
    elif(str(self.data)=='umount'):
      lustreSomething.umount()
    elif(str(self.data)=='createworkingdir'):
      lustreSomething.createworkingdir()


class LustreSomething:  
  """
  LustreSomething 클래스 부분에 필요한 함수를 구현한 다음에,
  MyTCPHandler 클래스 안에 handle 함수 부분에 구현한 함수를 추가.
  //LUSTREFS/server/0.1.0/package/script/test.py, subtools.py 두개 조합해서 SwitchToLustreFS (진행)
  1. SwitchToLustreFS (진행)
  2. SwitchToHdfs (예정)
  예제)
  def subprocess_open(slef,command):
      popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
      (stdoutdata, stderrdata) = popen.communicate()
      return stdoutdata, stderrdata 
  """
  def __init__(self):
    self.switchConf = SwitchConfig()

  def getHostsInfo(self):
    # local_hostname = socket.gethostname()
    # print "Local Hostname : " + local_hostname
    # ambari server hostname
    #self.subprocess_open('echo "getHostInfo0" >> /home/daemon/dm-log.info')
    config = ConfigParser.ConfigParser()
    #self.subprocess_open('echo "getHostInfo0" >> /home/daemon/dm-log.info')
    config.read("/etc/ambari-agent/conf/ambari-agent.ini")
    #self.subprocess_open('echo "getHostInfo1" >> /home/daemon/dm-log.info')
    global ambari_url
    ambari_url = config.get('server','hostname')
    print "Ambari server Hostname : " + ambari_url
    #self.subprocess_open('echo "getHostInfo2" >> /home/daemon/dm-log.info')
    # cluster_name
    headers = {
      'X-Requested-By': 'ambari',
    }
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    items = j["items"][0]["Clusters"]
    print "Cluster Name : " + items["cluster_name"]
    global cluster_name
    cluster_name = items["cluster_name"]
    #self.subprocess_open('echo "getHostInfo3" >> /home/daemon/dm-log.info')
    # LustreClient HostNames
    # curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/+cluster_name+/services/LUSTRE/components/LUSTRE_CLIENT
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/services/LUSTREMANAGER/components/LustreClient', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    result =[]
    for component in j["host_components"]:
      result.append(component["HostRoles"]["host_name"])
    #self.subprocess_open('echo "'+str(len(result))+'" >> /home/daemon/dm-log.info')
    return result


  def get_lustrefs_config(self):
    config = ConfigParser.ConfigParser()
    config.read("/etc/ambari-agent/conf/ambari-agent.ini")
    global ambari_url
    ambari_url = config.get('server','hostname')
    print "Ambari server Hostname : " + ambari_url

    # 클러스터 네임확인
    headers = {
      'X-Requested-By': 'ambari',
    }
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    items = j["items"][0]["Clusters"]
    print "Cluster Name : " + items["cluster_name"]
    global cluster_name
    cluster_name = items["cluster_name"]
    # LustreMDSMgmtService 호스트 추출

    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/services/LUSTREMANAGER/components/LustreMDSMgmtService', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    LustreMDSMgmtService_host = {
      "LustreMDSMgmtService_host" : j["host_components"][0]["HostRoles"]["host_name"]
    };

    # lustrefs-config-env 의 최근버전 추출
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'?fields=Clusters/desired_configs', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    lustrefs_config_env_tag = j['Clusters']['desired_configs']["lustrefs-config-env"]["tag"];
    # lustrefs-config-env 의 환경설정내용 추출
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/configurations?type=lustrefs-config-env&tag='+lustrefs_config_env_tag, headers=headers, auth=('admin', 'admin'));
    j = json.loads(r.text)

    lustrefs_config_env =  j["items"][0]["properties"];


    r = requests.get('http://'+ambari_url+':8080/views/Lustre_View/1.0.0/Lustre_View/api/v1/lustre/getClientFolder', headers=headers, auth=('admin', 'admin'))
    lustrefs_config_env['mount_client'] = r.text
    

    # lustre-default-config 의 최근버전 추출
    # r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'?fields=Clusters/desired_configs', headers=headers, auth=('admin', 'admin'))
    # j = json.loads(r.text)
    # lustre_default_config_tag = j['Clusters']['desired_configs']["lustre-default-config"]["tag"];
    
    # lustre-default-config 의 환경설정내용 추출
    # r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/configurations?type=lustre-default-config&tag='+lustre_default_config_tag, headers=headers, auth=('admin', 'admin'));
    # j = json.loads(r.text)

    # lustre_default_config = j["items"][0]["properties"];
    # self.subprocess_open('echo "lustre_default_config '+str(lustre_default_config)+'" >> /home/daemon/dm-log.info')
    # # lustre-default-config lustrefs-config-env LustreMDSMgmtService 병합
    # lustre_default_config.update(lustrefs_config_env);
    # lustre_default_config.update(LustreMDSMgmtService_host);
    # self.subprocess_open('echo "lustre_default_config '+str(lustre_default_config)+'" >> /home/daemon/dm-log.info')
    # return lustre_default_config;

    self.subprocess_open('echo "lustrefs_config_env '+str(lustrefs_config_env)+'" >> /home/daemon/dm-log.info')
    lustrefs_config_env.update(LustreMDSMgmtService_host);
    return lustrefs_config_env;

  def mount(self):
    lustrefs_config = self.get_lustrefs_config();
    #mdt_fsname = "mylustre"
    #mds_host = "master"
    self.subprocess_open('echo "lustre_default_config '+str(lustrefs_config)+'" >> /home/daemon/dm-log.info')
    mds_host = lustrefs_config["LustreMDSMgmtService_host"];
    self.subprocess_open('echo "'+mds_host+'" >> /home/daemon/dm-log.info')
    mdt_fsname = lustrefs_config["lustrefs_mountpoint"];
    self.subprocess_open('echo "'+mdt_fsname+'" >> /home/daemon/dm-log.info')
    
    lustre_client_folder = lustrefs_config["mount_client"];
    
    self.subprocess_open('echo "'+lustre_client_folder+'" >> /home/daemon/dm-log.info')

    #### mount ####
    check = commands.getstatusoutput('mount | grep '+mdt_fsname)
    if len(str(check[1])) == 0:      

      # for hostname in ['1.1.1.1','2.2.2.2']:
      hosts = self.getHostsInfo()
      # hosts = ['cn7','cn8','cn9','gpu']
      for hostname in hosts:            
        self.subprocess_open('echo "'+hostname+'" >> /home/daemon/dm-log.info')
        self.subprocess_open("ssh root@"+hostname+" -T \"mkdir -p "+lustre_client_folder+" \"")
        self.subprocess_open("ssh root@"+hostname+" -T \"mkdir -p /tmp2 \"")
        self.subprocess_open("ssh root@"+hostname+" -T \"mount -t lustre "+mds_host+"@tcp:/"+mdt_fsname+" "+lustre_client_folder+" > /tmp/mounthistory \"")

        self.subprocess_open("ssh root@"+hostname+" -T \"mkdir -p "+lustre_client_folder+"/hadoop \"")

        self.subprocess_open("ssh root@"+hostname+" -T \"mkdir -p "+lustre_client_folder+"/hadoop \"")
        self.subprocess_open("ssh root@"+hostname+" -T \"mkdir -p "+lustre_client_folder+"/hadoop/tmp2 \"")
        self.subprocess_open("ssh root@"+hostname+" -T \"chown hdfs. "+lustre_client_folder+"/hadoop/tmp2 \"")

        self.subprocess_open("ssh root@"+hostname+" -T \"chmod 777 "+lustre_client_folder+"/hadoop \"")
        print hostname;
        try:
          self.subprocess_open("ssh root@"+hostname+" -T \"mkdir -p "+lustre_client_folder+"/temp_disk/"+hostname+"\"")
        except Exception, e:
          raise e
        self.subprocess_open("ssh root@"+hostname+" -T \"mount -B "+lustre_client_folder+"/temp_disk/"+hostname+" /tmp2 > /tmp/mounthistory \"")
        
        self.subprocess_open("ssh root@"+hostname+" -T \" mkdir /tmp2/hadoop-yarn \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" chown -R yarn. /tmp2/hadoop-yarn \"")
        
        self.subprocess_open("ssh root@"+hostname+" -T \" mkdir /tmp2/yarn \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" mkdir /tmp2/yarn/local \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" mkdir /tmp2/yarn/log \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" mkdir /tmp2/yarn/timeline \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" chown -R yarn. /tmp2/yarn \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" chown yarn. /tmp2/yarn/local \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" chown yarn. /tmp2/yarn/log \"")
        self.subprocess_open("ssh root@"+hostname+" -T \" chown yarn. /tmp2/yarn/timeline \"")
        self.subprocess_open('echo "'+lustre_client_folder+'"0 >> /home/daemon/dm-log.info')
        #self.subprocess_open("ssh root@"+hostname+" -T \" mkdir –p "+lustre_client_folder+"/hadoop/tmp2 \"")
        self.subprocess_open('echo "'+lustre_client_folder+'"1 >> /home/daemon/dm-log.info')
        self.subprocess_open("ssh root@"+hostname+" -T \" chown hdfs. "+lustre_client_folder+"/hadoop/tmp2 \"")
        self.subprocess_open('echo "'+lustre_client_folder+'"2 >> /home/daemon/dm-log.info')
        self.subprocess_open("ssh root@"+hostname+" -T \" chmod 777 "+lustre_client_folder+"/hadoop/tmp2 \"")
        self.subprocess_open('echo "'+lustre_client_folder+'"3 >> /home/daemon/dm-log.info')
        self.subprocess_open("ssh root@"+hostname+" -T \" mkdir -p "+lustre_client_folder+"/hadoop/user/yarn \"")
        self.subprocess_open('echo "'+lustre_client_folder+'"4 >> /home/daemon/dm-log.info')
        self.subprocess_open("ssh root@"+hostname+" -T \" chown yarn. "+lustre_client_folder+"/hadoop/user/yarn \"")
        self.subprocess_open('echo "'+lustre_client_folder+'"5 >> /home/daemon/dm-log.info')

      self.subprocess_open("mkdir -p "+lustre_client_folder+"/hadoop/ats  &> /tmp/yarn")
      self.subprocess_open("mkdir -p "+lustre_client_folder+"/hadoop/ats/done  &> /tmp/yarn")
      self.subprocess_open("mkdir -p "+lustre_client_folder+"/hadoop/ats/acrive &> /tmp/yarn")
      self.subprocess_open("chown yarn. "+lustre_client_folder+"/hadoop/ats &> /tmp/yarn")
      self.subprocess_open("chown yarn. "+lustre_client_folder+"/hadoop/ats/acrive &> /tmp/yarn")
      self.subprocess_open("chown yarn. "+lustre_client_folder+"/hadoop/ats/done &> /tmp/yarn")

        # self.subprocess_open("ssh root@"+hostname+" -T \" mkdir -p /lustre/hadoop/ats/active \"")
        # self.subprocess_open("ssh root@"+hostname+" -T \" mkdir -p /lustre/hadoop/hdp/apps/2.6.3.0-235/mapreduce \"")
        # self.subprocess_open("ssh root@"+hostname+" -T \" yes | cp -r /usr/hdp/2.6.3.0-235/hadoop/* /lustre/hadoop/hdp/apps/2.6.3.0-235/mapreduce \"")
        
    else:
      print('already mount')
      self.subprocess_open('echo "already mount" >> /home/daemon/dm-log.info')
      hosts2 = self.getHostsInfo()
      #hosts2 = [u'slave1.hadoop.com', u'slave2.hadoop.com', u'slave3.hadoop.com']
      #self.subprocess_open('echo "getHostInfo" >> /home/daemon/dm-log.info')
      #self.subprocess_open('echo "'+str(len(hosts2))+'" >> /home/daemon/dm-log.info')
      for hostname1 in hosts2:
        self.subprocess_open('echo "'+hostname1+'" >> /home/daemon/dm-log.info')

  def umount(self):
    #### umount ####
    lustrefs_config = self.get_lustrefs_config();
    
    lustre_client_folder = lustrefs_config["mount_client"];


    self.subprocess_open('echo "'+lustre_client_folder+'" >> /home/daemon/dm-log.info')

    check = commands.getstatusoutput('mount | grep lustre')
    self.subprocess_open('echo "'+str(check)+'" >> /home/daemon/dm-log.info')
    if len(str(check[1])) == 0:
      print('not mount lustre')     

    else:
      hosts = self.getHostsInfo()
      # hosts = ['cn7','cn8','cn9','gpu']
      try:
        for hostname in hosts:
          self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l /tmp2 > /tmp/mounthistory \"")
          self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l "+lustre_client_folder+" > /tmp/mounthistory \"")
          self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l /tmp2 > /tmp/mounthistory \"")
          self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l "+lustre_client_folder+" > /tmp/mounthistory \"")

          print('success lustre unmount !')

      except:
        try:
          for hostname in hosts:
            self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l /tmp2 > /tmp/mounthistory \"")
            self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l "+lustre_client_folder+" > /tmp/mounthistory \"")
            self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l /tmp2 > /tmp/mounthistory \"")
            self.subprocess_open("ssh root@"+hostname+" -T \"nohup umount -l "+lustre_client_folder+" > /tmp/mounthistory \"")

            print('success lustre unmount by -l option')

        except Exception, e:
          print('not umount lustre!!')
          raise e
  def do_something(self):
      with open("/tmp/timecheck/current_time.txt", "w") as f:
          f.write("현재 시간 " + time.ctime() +"\n")
  
  def subprocess_open(slef,command):
    popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdoutdata, stderrdata) = popen.communicate()
    return stdoutdata, stderrdata

  def switchToLustreFS(self):
    self.switchConf.toLustrefs()

  def createworkingdir(self):
    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /ats  &> /tmp/yarn \" ")
    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /ats/active &>> /tmp/yarn \" ")

    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp &>> /tmp/yarn \" ")
    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp/apps &>> /tmp/yarn \" ")

    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp/apps/2.6.3.0-235 &>> /tmp/yarn \" ")
    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp/apps/2.6.3.0-235/mapreduce &>> /tmp/yarn \" ")
    # self.subprocess_open(" hadoop fs -put /usr/hdp/2.6.3.0-235/hadoop/mapreduce.tar.gz /hdp/apps/2.6.3.0-235/mapreduce &>> /tmp/yarn ")
    
    # self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mr-history &> /tmp/mapred \" ")
    # self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mr-history/done &>> /tmp/mapred \" ")
    # self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mr-history/tmp &>> /tmp/mapred \" ")

    # self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mapred &>> /tmp/mapred \" ")
    # self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mapred/system &>> /tmp/mapred \" ")


    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /ats  &> /tmp/yarn \" ")
    # self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /ats/active &>> /tmp/yarn \" ")

    self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp &>> /tmp/yarn \" ")
    self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp/apps &>> /tmp/yarn \" ")
    self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp/apps/2.6.3.0-235 &>> /tmp/yarn \" ")
    self.subprocess_open(" su -l yarn -c \"hadoop fs -mkdir /hdp/apps/2.6.3.0-235/mapreduce &>> /tmp/yarn \" ")
    self.subprocess_open(" hadoop fs -put /usr/hdp/2.6.3.0-235/hadoop/mapreduce.tar.gz /hdp/apps/2.6.3.0-235/mapreduce &>> /tmp/yarn ")
    self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mr-history &> /tmp/mapred \" ")
    self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mr-history/done &>> /tmp/mapred \" ")
    self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mr-history/tmp &>> /tmp/mapred \" ")
    self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mapred &>> /tmp/mapred \" ")
    self.subprocess_open(" su -l mapred -c \"hadoop fs -mkdir /mapred/system &>> /tmp/mapred \" ")

  def switchToHdfs(self):
    self.switchConf.toHdfs()


if __name__ == "__main__":
  #pid2 추가 
  daemon = MyDaemon('/tmp/daemon-lustre.pid')
  if len(sys.argv) == 2:
    if 'start' == sys.argv[1]:
      daemon.start()
    elif 'stop' == sys.argv[1]:
      daemon.stop()
    elif 'restart' == sys.argv[1]:
      daemon.restart()
    elif 'status' == sys.argv[1]:
      daemon.status()         
    else:
      print "Usage: %s start|stop|status|restart" % sys.argv[0]
      sys.exit(2)
      sys.exit(0)
else:
  print "Usage: %s start|stop|status|restart" % sys.argv[0]
  sys.exit(2)
