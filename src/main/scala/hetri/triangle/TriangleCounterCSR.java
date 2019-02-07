/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: TriangleCounterCSR.java
 * - Local triangle counting implementation for the CSR graph format.
 */

package hetri.triangle;


import hetri.graph.CSR;

public class TriangleCounterCSR {

    /**
     * count local triangles whose pivot edge is in `ij`, port edge is in `ik`, and starboard edge is in `jk`.
     * @param ij a graph of pivot edges.
     * @param ik a graph of port edges.
     * @param jk a graph of starboard edges.
     * @return
     */
    static public long countTriangles(CSR ij, CSR ik, CSR jk) {

        long count = 0;

        for(int u = 0; u < ij.numNodes; u++) {

            int end = ij.nodes_val[u + 1];

            for (int i = ij.nodes_val[u]; i < end; i++) {
                int v = ij.edges[i];

                count += countIntersection(u, v, ik, jk);
            }

        }

        return count;

    }

    /**
     * count the intersecting nodes of two neighbor sets
     * @param u the first node
     * @param v the second node
     * @param ik the graph including the first neighbor set
     * @param jk the graph icnluding the second neighbor set
     * @return the number of intersecting nodes
     */
    private static long countIntersection(int u, int v, CSR ik, CSR jk) {

        if (u >= ik.numNodes || v >= jk.numNodes) return 0;


        int cur_u = ik.nodes_val[u];
        int end_u = ik.nodes_val[u + 1];

        int cur_v = jk.nodes_val[v];
        int end_v = jk.nodes_val[v + 1];

        long count = 0;

        while(cur_u < end_u && cur_v < end_v){

            int val_u = ik.edges[cur_u];
            int val_v = jk.edges[cur_v];

            if(val_u < val_v){
                cur_u++;
            }
            else if(val_u > val_v){
                cur_v++;
            }
            else{
                count++;
                cur_u++;
                cur_v++;
            }
        }

        return count;

    }

}
