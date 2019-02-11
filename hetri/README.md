# HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters

* HeTri (Heterogeneous Triangle Enumeration) is a tool for enumerating all triangles in a large graph.
* Given an undirected simple graph, HeTri enumerates all triangles (three-node-cycles).
* Authors
    * Ha-Myung Park (hamyung.park@snu.ac.kr), Seoul National University
    * U Kang (ukang@snu.ac.kr), Seoul National University


## Build

`HeTri` uses SBT (Simple Build Tool) to manage dependencies and build the whole project. To build the project, type the following command in terminal:

```bash
tools/sbt assembly
```

## How to run `HeTri`

Hadoop and dependencies should be installed in your system in advance. The tested environment si as follows:

  * Java v1.8.0
  * Scala v2.11.4
  * Hadoop v2.7.3

Please refer the following code to run `HeTri`:

```bash

hadoop jar bin/hetri-0.1.jar hetri.HeTri \
                                       -Dallocator=$ALLOCATOR \
                                       -DnumColors=$NUM_COLORS \
                                       $DATA
```

Options:

  * `$ALLOCATOR` is the parallel scheduling algorithm of `HeTri` (options: `mlc` (default), `rand`, `greedy`).
  * `$NUM_COLOR` is the number of node colors.
  * `$DATA` is the input file path.



## Datasets

All the datasets used in this paper are publicly available.

| Dataset                | Nodes            | Edges           | Source                                                     |
|------------------------|------------------|-----------------|------------------------------------------------------------|
| LiveJournal (LJ)       | 4.8M             | 69M             | http://snap.stanford.edu                                   |
| Twitter (TWT)          | 42M              | 1.2B            | http://an.kaist.ac.kr/trace/WWW2010.html                   |
| Friendster (FS)        | 66M              | 1.8B            | http://webdatacommons.org/hyperlinkgraph                   |
| SubDomain (SD)         | 101M             | 1.9B            | http://webscope.sandbox.yahoo.com                          |
| YahooWeb (YW)          | 1.4B             | 6.6B            | http://boston.lti.cs.cmu.edu/clueweb09                     |
| ClueWeb09 (CW09)       | 4.8B             | 7.9B            | http://www.lemurproject.org/clueweb09/webGraph.php         |
| ClueWeb12 (CW12)       | 6.3B             | 72B             | http://www.lemurproject.org/clueweb12/webGraph.php         |
| RMAT-k (k=23, ..., 29) | 2<sup>k</sup>    | 2<sup>k+4</sup> | RMAT with parameter (a,b,c,d) = (0.57, 0.19, 0.19, 0.05)   |
