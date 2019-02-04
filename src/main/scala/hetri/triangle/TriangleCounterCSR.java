package hetri.triangle;


import hetri.graph.CSR;

public class TriangleCounterCSR {

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
