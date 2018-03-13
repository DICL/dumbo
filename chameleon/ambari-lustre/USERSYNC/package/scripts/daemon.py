#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, time, atexit
from signal import SIGTERM

class Daemon:

    def __init__(self, pidfile, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.pidfile = pidfile

    def daemonize(self):

        try:
            pid = os.fork()            
            if pid > 0:                
                sys.exit(0)
        except OSError, e:
            sys.stderr.write("Fork #1 failed: %d (%s)\n" % (e.errno, e.strerror))
            sys.exit(1)

        os.chdir("/")
        os.setsid()
        os.umask(0)

        
        try:
            pid = os.fork()
            if pid > 0:                
                sys.exit(0)
        except OSError, e:
            sys.stderr.write("Fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
            sys.exit(1)

        sys.stdout.flush()
        sys.stderr.flush()
        si = file(self.stdin, 'r')
        so = file(self.stdout, 'a+')
        se = file(self.stderr, 'a+', 0)
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        atexit.register(self.delpid)
        pid = str(os.getpid())
        file(self.pidfile,'w+').write("%s\n" % pid)

    def delpid(self):
        os.remove(self.pidfile)

    def start(self):

        try:
            pf = file(self.pidfile,'r')
            pid = int(pf.read().strip())
            pf.close()            
        except IOError:
            pid = None



        if pid:
            message = "[INFO] Daemon already running \n[INFO] Pidfile %s already exist.\n"
            sys.stderr.write(message % self.pidfile)
            sys.exit(1)
        elif not pid:
            os.system('systemctl start slapd')
            os.system('systemctl enable slapd ')
            startingMessage="[INFO] Daemon staring\n"
            sys.stdout.write(startingMessage)


        # Start the daemon
        self.daemonize()
        self.run()

    def stop(self):     

        try:
            pf = file(self.pidfile,'r')
            pid = int(pf.read().strip())
            pf.close()
            
        except IOError:
            pid = None

        if not pid:
            message = "[INFO] Daemon not running \n[INFO] Pidfile %s does not exist.\n"
            sys.stderr.write(message % self.pidfile)
            return # not an error in a restart
        elif pid:
            os.system('systemctl stop slapd')
            os.system('systemctl disable slapd ')
            stoppingMessage="[INFO] Daemon stopping\n"
            sys.stdout.write(stoppingMessage)

        # Try killing the daemon process
        try:
            while 1:
                os.kill(pid, SIGTERM)
                time.sleep(0.1)
        except OSError, err:
            err = str(err)
            if err.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print str(err)
                sys.exit(1)

    def restart(self):
        self.stop()       
        self.start()

    def status(self):
        try:
            pf = file(self.pidfile,'r')
            pid = int(pf.read().strip())
            pf.close()            
        except IOError:
            pid = None

        if pid:
            message = "[INFO] Daemon already running\n"
            sys.stdout.write(message)
        elif not pid:
            message = "[INFO] Daemon not running\n"
            sys.stdout.write(message)                  


    def run(self):
        """
        You should override this method when you subclass Daemon. It will be called after the process has been
        daemonized by start() or restart().
        """

    def install(self):
        """
        You should override this method when you subclass Daemon. It will be called after the process has been
        daemonized by start() or restart().
        """