/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: TriangleCounter.java
 * - Local triangle counting implementation for the CSRV graph format.
 */

package hetri.triangle;

import hetri.graph.CSRV;
import hetri.graph.Graph;

public class TriangleCounter {

    /**
     * count local triangles whose pivot edge is in `ij`, port edge is in `ik`, and starboard edge is in `jk`.
     * @param ij a graph of pivot edges.
     * @param ik a graph of port edges.
     * @param jk a graph of starboard edges.
     * @param parallel not used
     * @return the number of triangles
     */
    static public long countTriangles(Graph ij, Graph ik, Graph jk, boolean parallel) {

        CSRV ij_csrv = (CSRV) ij;
        CSRV ik_csrv = (CSRV) ik;
        CSRV jk_csrv = (CSRV) jk;

        long count = 0;

        for(int i = 0; i < ij_csrv.numNodes; i++){

            int startPos = ij_csrv.nodes_val[i];
            int endPos = ij_csrv.nodes_val[i+1];

            int u = ij_csrv.nodes_id[i];

            int idx_u = ik_csrv.getIdxOf(u);

            if(idx_u < 0) continue;

            int startPos_u = ik_csrv.nodes_val[idx_u];
            int endPos_u = ik_csrv.nodes_val[idx_u + 1];

            for(int j = startPos; j < endPos; j++){
                int v = ij_csrv.edges[j];

                int idx_v = jk_csrv.getIdxOf(v);

                if(idx_v < 0) continue;

                int startPos_v = jk_csrv.nodes_val[idx_v];
                int endPos_v = jk_csrv.nodes_val[idx_v + 1];

                count += countIntersect(ik_csrv, jk_csrv, startPos_u, endPos_u, startPos_v, endPos_v);

            }


        }

        return count;

    }

    /**
     * count the intersecting nodes of two neighbor sets
     * @param ik the graph including the first neighbor set
     * @param jk the graph including the second netobor set
     * @param cur_u the start position index of the first neighbor set
     * @param end_u the end position index of the first neighbor set (exclusive)
     * @param cur_v the start position index of the second neighbor set
     * @param end_v the end position index of the second neighbor set (exclusive)
     * @return the number of the intersecting nodes
     */
    private static long countIntersect(CSRV ik, CSRV jk, int cur_u, int end_u, int cur_v, int end_v) {

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
