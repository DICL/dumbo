# Debugging option
DEBUG = True

# Resource types
RESOURCE_CORE = 'RESOURCE_CORE'

# Core: 0~15 (16-core per socket)
coreNumMax = 15

# Maximum DRAM bandwidth. It is measured using STREAM benchmark (thread=16)
DRAM_MAX = 30000

# Total number of NUMA node in the system
nodeNum = 4

# It assumes that LC and Batch are working on the `targetNode`
# The controller manages the resources on the `targetNode`
targetNode = 1

# Controller writes load and latency log file in this path
SCRIPT_ROOT = "./"

LC_STAT_PATH = "/usr/src/memcached/memcached_client/stat.txt"

loadGeneratorName = 'mh-dc-client'

# Minimum resource allocation counts
minCoreAllocCount = 1
lcMinCore = 4

# We don't control the resource of load generator
# Ignore load generator container
blackList = [loadGeneratorName]

# Accept containers with specific prefix
whitePrefix = 'mh-'
