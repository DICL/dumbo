package hetri.type;

import org.apache.hadoop.io.ShortWritable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class BytePairWritable implements WritableComparable<BytePairWritable> {

    private byte u, v;

    public BytePairWritable() {}

    public BytePairWritable(byte u, byte v) {
        set(u, v);
    }

    public void set(byte u, byte v) {
        this.u = u;
        this.v = v;
    }

    public int get_u() {
        return u;
    }

    public int get_v() {
        return v;
    }

    @Override
    public int compareTo(BytePairWritable o) {
        if(this.u != o.u) return Byte.compare(this.u, o.u);
        else return Byte.compare(this.v, o.v);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeByte(u);
        out.writeByte(v);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        u = in.readByte();
        v = in.readByte();
    }

    static { // register default comparator
        WritableComparator.define(BytePairWritable.class, new ShortWritable.Comparator());
    }
}
