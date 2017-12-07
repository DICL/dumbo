#!/usr/bin/env python
import signal
import sys
import threading
import time
from datetime import datetime

from srm import cgutils, resutils
from srm.resutils import coreAllocTable
from srm.settings import *

# Global lock
lock = threading.Lock()

# Init global variables
flagBatchGrow = False
latencyLogFile = open(SCRIPT_ROOT + '/latencyLog.txt', 'w')
loadLogFile = open(SCRIPT_ROOT + '/loadLog.txt', 'w')


# Gracefully exit when got signal
def exit_handler(signal, frame):
    latencyLogFile.close()
    loadLogFile.close()
    sys.exit(0)

# Register SIGINT signal handler
signal.signal(signal.SIGINT, exit_handler)


def getMemoryBandwidth(socket=0):
    with open(SCRIPT_ROOT + '/pcm.log') as f:
        last = f.readlines()[-1]
        dataList = last.split(";")
        bandwidth = dataList[13].strip()
    return float(bandwidth)


class SubController(threading.Thread):
    def __init__(self, debug=False, sleepTime=2, dramLimitRatio=0.8):
        self.PHASE_CORE = 'PHASE_CORE'
        self.DRAM_LIMIT = DRAM_MAX * dramLimitRatio
        self.phase = self.PHASE_CORE
        self.sleepTime = sleepTime
        self.debug = debug
        threading.Thread.__init__(self)

    def log(self, fmt):
        if self.debug:
            print("[SUBC] {}".format(fmt))

    def run(self):
        self.phase = self.PHASE_CORE
        global flagBatchGrow

        while True:
            # Acquire global lock
            lock.acquire()

            # Get available resources
            availCores = resutils.getLC(coreAllocTable) - lcMinCore
            availResources = availCores

            # Check memory bandwidth limit
            membw = getMemoryBandwidth(socket=targetNode)
            self.log("Memory bandwidth: {}".format(membw))
            if membw > self.DRAM_LIMIT:
                self.log("Over DRAM BW limit! Memory BW: {} / LIMIT: {}".format(
                    membw, self.DRAM_LIMIT))
                self.log("Realloc CORE from batch to LC")
                resutils.reallocResource(RESOURCE_CORE, enabledBatchCGs, enabledLCCGs, 1)
                lock.release()
                time.sleep(self.sleepTime)
                continue

            self.log("flagBatchGrow: {} / availResources: {}".format(flagBatchGrow, availResources))
            if not flagBatchGrow or availResources == 0:
                lock.release()
                time.sleep(self.sleepTime)
                continue

            allCGs = cgutils.getAllCGs()
            enabledBatchCGs = cgutils.getEnabledBatchCGs(allCGs)
            enabledLCCGs = cgutils.getEnabledLCCGs(allCGs)

            if self.phase == self.PHASE_CORE:
                if availCores > 0:
                    self.log("Realloc core from LC to batch")
                    resutils.reallocResource(RESOURCE_CORE, enabledLCCGs, enabledBatchCGs, 1)
            else:
                raise RuntimeError("Invalid phase.")

            availCores = resutils.getLC(coreAllocTable) - lcMinCore
            if availCores < 3:
                self.log("Self disbaled! availCores: {}".format(availCores))
                flagBatchGrow = False

            lock.release()
            time.sleep(self.sleepTime)


def initAllocTable():
    allCGs = cgutils.getAllCGs()
    lcCG = cgutils.getLCCG(allCGs)
    enabledBatchCGs = cgutils.getEnabledBatchCGs(allCGs)

    # LC should exist
    assert lcCG
    
    # Enable LC
    lcCG.enable()
    resutils.allocAllResource(lcCG)

    if enabledBatchCGs:
        #  randomly chosen
        batchCG = enabledBatchCGs[0]
        resutils.reallocMinResource(lcCG, batchCG)

        for CG in enabledBatchCGs[1:]:
            CG.disable()


# Main controller, make high-level decision
class MainController(object):
    def __init__(self, target=1, sleepTime=15, subSleepTime=2, maxLoad=450000.0,
            safe_guard=0.3, debug=False, dramLimitRatio=0.8):
        self.debug = debug
        self.target = target
        self.maxLoad = maxLoad
        self.safe_guard = safe_guard
        self.sleepTime = sleepTime
        self.subSleepTime = subSleepTime
        self.dramLimitRatio = dramLimitRatio
        self.loadGenerator = cgutils.getLoadGenerator()

    def getStat(self):
        rv = self.loadGenerator.exec_run(
                'cat {}'.format(LC_STAT_PATH)).decode('ascii')
        return rv.strip()

    def getLoad(self, stat):
        load = float(stat.split(',')[0])
        return load/self.maxLoad
        
    def getLatency(self, stat):
        return float(stat.split(',')[1])

    def log(self, fmt):
        if self.debug:
            print("[MAIN] {}".format(fmt))

    def monitor(self):
        """
        Monitoring mode. Just monitor the status of latency-critical container
        without controlling hardware resources.
        """
        while True:
            stat = self.getStat()
            latency = self.getLatency(stat)
            load = self.getLoad(stat)

            slack = float(self.target - latency) / self.target
            timestamp=str(datetime.now())
            latencyLogFile.write("{},{}\n".format(timestamp, latency))
            loadLogFile.write("{},{}\n".format(timestamp, load))
            self.log("Load: {} / Latency: {} / Slack: {}".format(load, latency, slack))
            time.sleep(self.sleepTime)

    def run(self):
        global flagBatchGrow
        flagBatchGrow = False

        # Initialize allocTable and enable LC and disable Batch
        initAllocTable()

        # Start subcontroller
        subcontroller = SubController(
                debug=self.debug,
                sleepTime=self.subSleepTime,
                dramLimitRatio=self.dramLimitRatio)
        subcontroller.start()

        self.log("MainController - maxLoad: {}, sleepTime: {}, lcMinCore: {}".format(
            self.maxLoad, self.sleepTime, lcMinCore))
        while True:
            # Acquire global lock
            lock.acquire()

            # Get current status of latency-critical workload
            # i.e., latency and load
            stat = self.getStat()
            latency = self.getLatency(stat)
            load = self.getLoad(stat)
            slack = float(self.target - latency) / self.target

            timestamp=str(datetime.now())
            latencyLogFile.write("{},{}\n".format(timestamp, latency))
            loadLogFile.write("{},{}\n".format(timestamp, load))
            self.log("{} / Load: {} / Latency: {} / Slack: {}".format(
                str(datetime.now()), load, latency, slack))

            allCGs = cgutils.getAllCGs()
            enabledBatchCGs = cgutils.getEnabledBatchCGs(allCGs)
            disabledBatchCGs = cgutils.getDisabledBatchCGs(allCGs)
            enabledLCCGs = cgutils.getEnabledLCCGs(allCGs)

            # Support only one LC
            assert len(enabledLCCGs) == 1
            lcCG = enabledLCCGs[0]

            if slack < 0:
                self.log("Negative slack, disable all batchCGs and get all resource from them")
                flagBatchGrow = False
                resutils.disableAllBatchCGs(lcCG, enabledBatchCGs)

            elif slack <= self.safe_guard or load > 0.85:
                self.log("Safe guard, disable batch grow")
                flagBatchGrow = False
                if enabledBatchCGs:
                    self.log("Safe guard, realloc resource from batch to LC")
                    resutils.reallocResource(RESOURCE_CORE, enabledBatchCGs, enabledLCCGs, 1)
                else:
                    self.log("Safe guard, but there is no enabled BatchCGs")

            elif load < 0.80:
                self.log("Good LC load is low")
                if disabledBatchCGs:
                    flagBatchGrow = False
                    batchCGToEnable = disabledBatchCGs[0]

                    self.log("Disabled batch exist. Enable it".format(batchCGToEnable))
                    resutils.reallocMinResource(lcCG, batchCGToEnable)
                    batchCGToEnable.enable()

                elif enabledBatchCGs:
                    self.log("There is no disabled batch, enable batch grow")
                    flagBatchGrow = True

                else:
                    # There is no batch
                    flagBatchGrow = False

            lock.release()
            time.sleep(self.sleepTime)


if __name__ == '__main__':
    MainController(debug=DEBUG).run()
