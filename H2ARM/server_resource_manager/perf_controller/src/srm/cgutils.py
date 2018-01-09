import docker

from srm.model import ContainerGroup
from srm.settings import *

client = docker.from_env()

def getLoadGenerator():
    return client.containers.get(loadGeneratorName)


def getAllCGs():
    containers = client.containers.list()
    allCGs = list()
    nBatchContainers = list()

    # Create ContainerGroups
    for c in containers:
        # If container is in the black list, ignore it.
        if c.name in blackList:
            continue
        
        # If container name is not start with specific prefix, ignore it
        if not c.name.startswith(whitePrefix):
            continue

        # Get metadata of containers
        # i.e., Latency-Critical/Batch
        latencyCritical = c.attrs['Config']['LatencyCritical']

        if not latencyCritical:
            nBatchContainers.append(c)
        else:
            cg = ContainerGroup(
                containers=[c],
                latencyCritical=latencyCritical,
            )
            allCGs.append(cg)

    # Handling batch container groups
    if nBatchContainers:
        cg = ContainerGroup(
            containers=nBatchContainers,
            latencyCritical=False,
        )
        allCGs.append(cg)

    return allCGs


def getEnabledBatchCGs(allCGs):
    return [x for x in allCGs if x.enabled and not x.latencyCritical]


def getDisabledBatchCGs(allCGs):
    return [x for x in allCGs if not x.enabled and not x.latencyCritical]


def getEnabledLCCGs(allCGs):
    enabledLCCGs = [x for x in allCGs if x.enabled and x.latencyCritical]
    assert(len(enabledLCCGs) == 1)
    assert(len(enabledLCCGs[0].containers) == 1)
    return enabledLCCGs


def getLCCG(allCGs):
    enabledLCCGs = getEnabledLCCGs(allCGs)
    return enabledLCCGs[0]
