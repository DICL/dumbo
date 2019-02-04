package hetri.triangle;

import hetri.graph.CSRV;
import hetri.graph.Graph;

import java.util.Iterator;

public class TriangleCounter {

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

    static public long countIntersect(Iterator<Integer> uN, Iterator<Integer> vN) {

        long count = 0;


        int u_head = uN.hasNext() ? uN.next() : -1;
        int v_head = vN.hasNext() ? vN.next() : -1;

        while(u_head >= 0 && v_head >= 0){
            if(u_head < v_head){
                u_head = uN.hasNext() ? uN.next() : -1;
            }
            else if(u_head > v_head){
                v_head = vN.hasNext() ? vN.next() : -1;
            }
            else{
                count++;
                u_head = uN.hasNext() ? uN.next() : -1;
                v_head = vN.hasNext() ? vN.next() : -1;
            }
        }

        return count;


    }

}
