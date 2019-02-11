/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: IntPairWritable.java
 * - Writable for an int pair.
 */

package hetri.type;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class IntPairWritable extends IntPair implements WritableComparable<IntPair> {

    public IntPairWritable() {}

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(get_u());
        out.writeInt(get_v());
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        set_u(in.readInt());
        set_v(in.readInt());
    }

    static { // register default comparator
        WritableComparator.define(IntPairWritable.class, new LongWritable.Comparator());
    }
}
