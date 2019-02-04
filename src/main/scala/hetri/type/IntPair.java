package hetri.type;

import java.util.Objects;

public class IntPair implements Comparable<IntPair>{
    private int u, v;

    public IntPair(){}

    public IntPair(int u, int v){
        set(u, v);
    }

    public int get_u() {
        return u;
    }

    public int get_v() {
        return v;
    }

    public void set_u(int u) {
        this.u = u;
    }

    public void set_v(int v) {
        this.v = v;
    }

    public void set(int u, int v){
        this.u = u;
        this.v = v;
    }

    static public long toLong(int u, int v){
        return (((long) u) << 32) | v;
    }

    public long getAsLong() {
        return toLong(u, v);
    }

    @Override
    public String toString() {
        return "(" + u + ", " + v + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        IntPair intPair = (IntPair) o;
        return u == intPair.u &&
                v == intPair.v;
    }

    @Override
    public int hashCode() {
        return Objects.hash(u, v);
    }

    @Override
    public int compareTo(IntPair o) {
        return Long.compare(this.getAsLong(), o.getAsLong());
    }
}
