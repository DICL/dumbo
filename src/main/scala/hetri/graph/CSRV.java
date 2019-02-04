package hetri.graph;

import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import hetri.type.IntPair;
import it.unimi.dsi.fastutil.longs.LongArrayList;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.Iterator;

public class CSRV implements Graph {

    Logger logger = Logger.getLogger(getClass());

    private static double BACKUP_RATIO = 1.2;

    public int numEdges = 0;
    public int numNodes = 0;
    public int[] edges, nodes_val, nodes_id;

    public CSRV(){}

    public CSRV(Path pedge, Path pnode, FileSystem fs) throws IOException {
        read(pedge, pnode, fs);
    }

    public CSRV(Input iedge, Input inode){
        read(iedge, inode);
    }

    public Iterator<Integer> nodeIterator(){
        return Arrays.stream(nodes_id).iterator();
    }

    public int getIdxOf(int n) {
        return Arrays.binarySearch(nodes_id, 0, numNodes, n);
    }


    public void writeEdges(Output out) {
        out.writeInt(numEdges);

        for(int i = 0; i < numEdges; i++)
            out.writeInt(edges[i]);
    }

    public void writeNodes(Output out) {
        out.writeInt(numNodes);

        for(int i = 0; i < numNodes; i++) {
            out.writeInt(nodes_id[i]);
            out.writeInt(nodes_val[i]);
        }

        out.writeInt(nodes_val[numNodes]);
    }

    @Override
    public void write(Output out_edge, Output out_node) {
        writeEdges(out_edge);
        writeNodes(out_node);
    }

    private void _read(Input iedge, Input inode) {
        readEdges(iedge);
        readNodes(inode);
    }

    @Override
    public void read(Input iedge, Input inode) {
        _read(iedge, inode);
        iedge.close();
        inode.close();
    }

    @Override
    public void read(Path pedge, Path pnode, FileSystem fs) throws IOException {
        Input iedge = new Input(fs.open(pedge));
        Input inode = new Input(fs.open(pnode));

        _read(iedge, inode);

        iedge.close();
        inode.close();
    }


    private void readEdges(Input in) {
        if(in == null) return;

        int numEdges_new = in.readInt();

        if(edges == null || numEdges_new > edges.length){
            int sizeBefore = edges == null ? 0 : edges.length;
            int sizeAfter = (int) (numEdges_new * BACKUP_RATIO);
            logger.info("load new edges size from " + sizeBefore + "to " + sizeAfter);
            edges = new int[sizeAfter];
        }

        for(int i = 0; i < numEdges_new; i++){
            edges[i] = in.readInt();
        }

        numEdges = numEdges_new;
    }

    private void readNodes(Input in) {
        if(in == null) return;

        int numNodes_new = in.readInt();

        if(nodes_id == null || numNodes_new > nodes_id.length){
            int sizeBefore = nodes_val == null ? 0 : nodes_val.length;
            int sizeAfter = (int) (numNodes_new * BACKUP_RATIO) + 1;
            logger.info("load new nodes size from " + sizeBefore + "to " + sizeAfter);
            nodes_id = new int[sizeAfter];
            nodes_val = new int[sizeAfter + 1];
        }

        for(int i = 0; i < numNodes_new; i++){
            nodes_id[i] = in.readInt();
            nodes_val[i] = in.readInt();
        }

        nodes_val[numNodes_new] = in.readInt();

        numNodes = numNodes_new;
    }

    @Override
    public void writeFrom(Iterable<? extends IntPair> edges, Output oedge, Output onode) {
        LongArrayList arr = new LongArrayList();

        for(IntPair pair : edges){
            arr.add(pair.getAsLong());
        }

        arr.sort(null);

        int prev = -1;

        int len = arr.size();
        int numNodes = 0;
        for(int i = 0; i < len; i++){
            int u = (int) (arr.getLong(i) >> 32);

            if(u != prev){
                numNodes++;
                prev = u;
            }
        }

        onode.writeInt(numNodes);
        oedge.writeInt(len);

        prev = -1;
        for(int i = 0; i < len; i++){
            long e = arr.getLong(i);
            int u = (int) (e >> 32);
            int v = (int) e;

            if(u != prev){
                onode.writeInt(u);
                onode.writeInt(i);

                prev = u;
            }

            oedge.writeInt(v);
        }

        onode.writeInt(len);

        oedge.close();
        onode.close();
    }
}
