# -*- coding: utf-8 -*-
import commands
import sys, os, subprocess, ConfigParser, time, random, commands, socket, json, SocketServer
from socket import *
from select import *
from daemon_yarn import Daemon
from subtools import SubTools
#from switch_config import SwitchConfig
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
      time.sleep(1)      
      HOST, PORT = "localhost", 5690
      server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)
      server.serve_forever()

class MyTCPHandler(SocketServer.BaseRequestHandler):
  def handle(self):

    self.data = self.request.recv(1024).strip()
    YarnJobMonitor = YarnJobMonitor()        
    if(str(self.data)=='tolustrefs'):
      YarnJobMonitor.switchToLustreFS()
    elif(str(self.data)=="tohdfs"):
      YarnJobMonitor.switchToHdfs() #예정.      
    elif(str(self.data)=='sample02'):
      YarnJobMonitor.subprocess_open('ls -al /home/daemon > /home/daemon/ls.txt')
    elif(str(self.data)=='sample-time'):
      YarnJobMonitor.do_something()
    elif(str(self.data)=='mount'):
      YarnJobMonitor.mount()
    elif(str(self.data)=='umount'):
      YarnJobMonitor.umount()


class YarnJobMonitor:  
  """
  YarnJobMonitor 클래스 부분에 필요한 함수를 구현한 다음에,
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
    config = ConfigParser.ConfigParser()
    config.read("/etc/ambari-agent/conf/ambari-agent.ini")
    global ambari_url
    ambari_url = config.get('server','hostname')
    print "Ambari server Hostname : " + ambari_url
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

    # LustreClient HostNames
    # curl -u admin:admin http://192.168.1.194:8080/api/v1/clusters/+cluster_name+/services/LUSTRE/components/LUSTRE_CLIENT
    r = requests.get('http://'+ambari_url+':8080/api/v1/clusters/'+cluster_name+'/services/YARNJOBMONITORSERVICE/components/YARNJobMonitorClient', headers=headers, auth=('admin', 'admin'))
    j = json.loads(r.text)
    result =[]
    for component in j["host_components"]:
      result.append(component["HostRoles"]["host_name"])
    
    return result

  def do_something(self):
      with open("/tmp/timecheck/current_time.txt", "w") as f:
          f.write("현재 시간 " + time.ctime() +"\n")
  
  def subprocess_open(slef,command):
    popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    (stdoutdata, stderrdata) = popen.communicate()
    return stdoutdata, stderrdata

if __name__ == "__main__":
  #pid2 추가 
  daemon = MyDaemon('/tmp/daemon-yarnjobmonitor.pid')
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
