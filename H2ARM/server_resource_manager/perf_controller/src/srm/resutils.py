from collections import Counter
from functools import reduce

from srm.model import ContainerGroup
from srm.settings import *

# Error message
INVALID_RESOURCE_ERROR = "Invalid resource type."

# Resource allocation table
# list idx -> resource id (i.e., core number)
# list val -> resource owner (i.e., container group)
coreAllocTable = [None for x in range(coreNumMax+1)]


def getCoreString(coreList):
    coreBase = targetNode * (coreNumMax + 1) 
    newCoreList = ','.join(str(coreBase + x) for x in coreList)
    return newCoreList


def minResource(type, CG):
    if type == RESOURCE_CORE:
        if CG.latencyCritical:
            return lcMinCore
        else:
            return minCoreAllocCount
    else:
        raise RuntimeError(INVALID_RESOURCE_ERROR)


def reduceCountLC(x, y):
    if y and y.latencyCritical:
        return x + 1
    else:
        return x


def getLC(allocTable):
    return reduce(reduceCountLC, allocTable, 0)


def getAllocTable(type):
    if type == RESOURCE_CORE:
        allocTable = coreAllocTable
    else:
        raise RuntimeError(INVALID_RESOURCE_ERROR)
    return allocTable


def findIndices(lst, condition):
    return [idx for idx, elem in enumerate(lst) if condition(elem)]


def createCpuset(CG):
    coreList = findIndices(coreAllocTable, lambda x: x == CG)
    assert coreList
    return getCoreString(coreList)


def getAllocTableCounter(type):
    return Counter(getAllocTable(type))


def getCGwithMaxResource(type, CGs):
    allocTableCounter = getAllocTableCounter(type)
    maxCG = CGs[0]
    maxCount = allocTableCounter.get(maxCG, 0)

    for CG in CGs:
        count = allocTableCounter.get(CG, 0)
        if count > maxCount:
            maxCG = CG
            maxCount = count

    return maxCG


def getCGwithMinResource(type, CGs):
    allocTableCounter = getAllocTableCounter(type)
    minCG = CGs[0]
    minCount = allocTableCounter.get(minCG, 0)

    for CG in CGs:
        count = allocTableCounter.get(CG, 0)
        if count < minCount:
            minCG = CG
            minCount = count

    return minCG


def resourceCount(type, CG):
    allocTableCounter = getAllocTableCounter(type)
    return allocTableCounter.get(CG, 0)


def updateAllResource(CG):
    CG.update(RESOURCE_CORE, createCpuset(CG))


def updateResource(type, CG):
    if type == RESOURCE_CORE:
        value = createCpuset(CG)
    else:
        raise RuntimeError(INVALID_RESOURCE_ERROR)
    # Call Docker update API to affect to the real-world
    CG.update(type, value)


def removeResource(type, CG):
    allocTable = getAllocTable(type)
    resourceNum = None

    if type == RESOURCE_CORE:
        for resourceIdx, ownerCG in enumerate(allocTable):
            if ownerCG == CG:
                resourceNum = resourceIdx
                break
    else:
        raise RuntimeError(INVALID_RESOURCE_ERROR)

    # Requested resource does not exist for CG (ERROR)
    assert resourceNum is not None

    # Remove owner of this resource
    allocTable[resourceNum] = None

    # Call Docker update API to affect to the real-world
    # The resource is really removed after this line
    updateResource(type, CG)
    return resourceNum


def setCoreOwner(coreNum, newOwnerCG):
    allocTable = getAllocTable(RESOURCE_CORE)
    allocTable[coreNum] = newOwnerCG
    updateResource(RESOURCE_CORE, newOwnerCG)


def addCore(toCG, coreNum):
    ownerCG = coreAllocTable[coreNum]
    setCoreOwner(coreNum, toCG)


def reallocResource(type, fromCGs, toCGs, N):
    while N > 0:
        fromCG = getCGwithMaxResource(type, fromCGs)

        # Can't remove resource if CG have already less than minimum resource
        if resourceCount(type, fromCG) <= minResource(type, fromCG):
            break
        
        resourceNum = removeResource(type, fromCG)
        toCG = getCGwithMinResource(type, toCGs)

        if type == RESOURCE_CORE:
            addCore(toCG, resourceNum)
        else:
            raise RuntimeError(INVALID_RESOURCE_ERROR)
        N -= 1


def allocAllResource(CG):
    """Bulk resource update, give all resources to the given CG"""
    for coreNum, ownerCG in enumerate(coreAllocTable):
        coreAllocTable[coreNum] = CG
    updateAllResource(CG)


def reallocMinResource(fromCG, toCG):
    toMinCore = minResource(RESOURCE_CORE, toCG)
    coreRequirement = minResource(RESOURCE_CORE, fromCG) + toMinCore 
    if resourceCount(RESOURCE_CORE, fromCG) < coreRequirement:
        raise RuntimeError("Number of core is not enough")

    for i in range(toMinCore):
        coreNum = removeResource(RESOURCE_CORE, fromCG)
        addCore(toCG, coreNum)
       

def disableAllBatchCGs(lcCG, enabledBatchCGs):
    # Before reallocating resources disable all batch CGs
    for batchCG in enabledBatchCGs:
        batchCG.disable()

    allocAllResource(lcCG)
