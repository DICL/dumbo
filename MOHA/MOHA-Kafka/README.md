# This is source code repository for MOHA-Kafka

MOHA (Many-task computing On HAdoop) is a new framework to effectively combine Many-Task Computing technologies with the existing Big Data platform Hadoop. MOHA is developed as one of Hadoop YARN applications by utilizing YARN APIs so that it can transparently cohost
existing MTC applications with other Big Data processing frameworks such as MapReduce in a single Hadoop cluster. 

Users can utilize the MOHA Client (a plain Java object) to submit and monitor their MTC applications. Then MOHA runtime system automatically performs resource allocations through communicating with YARN ResourceManager and efficiently processes many tasks by employing multi-level
scheduling mechanism.