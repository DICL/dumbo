# -*- coding: utf-8 -*-
import os
import subprocess
import socket

class SubTools:
    def __init__(self):
        pass

    def makeDirectory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def xmlFileWriter(self,fileName,data):
        f = open(fileName, 'w')
        f.write(data)
        f.close()

    def xmlChanger(self,baseNodes, updNodes):
        for uNode in updNodes.findAll('property'):
            uName = uNode.find('name').string
            uValue = uNode.find('value').string

            for bNode in baseNodes.findAll('property'):
                if uName == bNode.find('name').string:
                    bNode.find('value').string = uValue

        return baseNodes

    def xmlFileReader(self,fileName):
        f = open(fileName, 'r')
        siteXml=""
        while True:
            line = f.readline()
                    
            if not line:
                break
            siteXml+=line
        f.close()    
        return siteXml   

    def sp_open(self,command):
        popen = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        (stdoutdata, stderrdata) = popen.communicate()
        return stdoutdata, stderrdata

    def excuteDaemon(self,option,port):
        # 5678 -> 5679
        HOST, PORT = "localhost", 5679
        PORT = port
        # data = " ".join(sys.argv[1:])
        data = option

        # Create a socket (SOCK_STREAM means a TCP socket)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            # Connect to server and send data
            sock.connect((HOST, PORT))
            sock.sendall(data + "\n")
            print data

            # Receive data from the server and shut down
            received = sock.recv(1024)
        finally:
            sock.close()

        print "Call_Target: {0}".format(data)
        if("Result: {0}".format(received)!=''):
            print "Result: {0} - SUCCESS".format(received)
        else:
            print "Result: FAIL"