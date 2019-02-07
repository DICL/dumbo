/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: Graph.java
 * - Interface for Graph formats.
 */


package hetri.graph;

import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import hetri.type.IntPair;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.util.Iterator;

public interface Graph {
    void write(Output out_edge, Output out_node);
    void read(Input in_edge, Input in_node);
    void read(Path pedge, Path pnode, FileSystem fs) throws IOException;
    void writeFrom(Iterable<? extends IntPair> edges, Output oedge, Output onode);
}
