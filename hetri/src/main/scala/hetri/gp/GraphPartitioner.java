/*
 * HeTri: Multi-level Node Coloring for Efficient Triangle Enumeration on Heterogeneous Clusters
 * Authors: Ha-Myung Park and U Kang
 *
 * -------------------------------------------------------------------------
 * File: GraphPartitioner.java
 * - The graph partitioning step of HeTri.
 */

package hetri.gp;

import com.esotericsoftware.kryo.io.Output;
import hetri.graph.CSR;
import hetri.graph.CSRV;
import hetri.graph.Graph;
import hetri.type.BytePairWritable;
import hetri.type.IntPairWritable;
import org.apache.hadoop.conf.Configurable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.StringTokenizer;

public class GraphPartitioner extends Configured implements Tool {

    /**
     * the main entry point
     * @param args
     * [0]: input file path, [1]: output file path
     * Tool runner parameters:
     * -D numColors the number of node colors
     * -D inputFormat seq: sequence file format, tsv: tab seperated edge list file format
     * @throws Exception by Hadoop
     */
    public static void main(String[] args) throws Exception {
        ToolRunner.run(new GraphPartitioner(), args);
    }

    /**
     * Submit the hadoop job
     * @param args
     * [0]: input file path, [1]: output file path
     * Tool runner parameters:
     * -D numColors the number of node colors
     * -D inputFormat seq: sequence file format, tsv: tab seperated edge list file format
     * @return 0
     * @throws Exception by Hadoop
     */
    @Override
    public int run(String[] args) throws Exception {

        Configuration conf = getConf();

        String input = args[0];
        String output = args[1];

        // the number of node colors
        int c = conf.getInt("numColors", 0);
        String inputFormat = conf.get("inputFormat", "text");

        conf.set("outputPath", output);

        Job job = Job.getInstance(conf, "[GP]" + input + "," + c);
        job.setJarByClass(getClass());

        if(inputFormat.equals("seq")){
            job.setMapperClass(GPMapperSeq.class);
            job.setInputFormatClass(SequenceFileInputFormat.class);
        }
        else{
            job.setMapperClass(GPMapper.class);
        }

        job.setPartitionerClass(GPPartitioner.class);
        job.setReducerClass(GPReducer.class);

        job.setMapOutputKeyClass(BytePairWritable.class);
        job.setMapOutputValueClass(IntPairWritable.class);

        job.setOutputFormatClass(NullOutputFormat.class);
        FileInputFormat.addInputPath(job, new Path(input));

        job.waitForCompletion(true);

        return 0;
    }

    public static class GPMapper extends Mapper<Object, Text, BytePairWritable, IntPairWritable>{

        // the number of node colors
        int c;

        /**
         * Setup before execution
         * @param context of Hadoop
         */
        @Override
        protected void setup(Context context) {
            this.c = context.getConfiguration().getInt("numColors", 0);
        }

        BytePairWritable ok = new BytePairWritable();
        IntPairWritable ov = new IntPairWritable();

        /**
         * Parse an edge and emit the edge by its color.
         * @param key not used
         * @param value an edge in a tab separated text format
         * @param context of Hadoop
         * @throws IOException of Hadoop
         * @throws InterruptedException of Hadoop
         */
        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            StringTokenizer st = new StringTokenizer(value.toString());

            long u = Long.parseLong(st.nextToken());
            long v = Long.parseLong(st.nextToken());
            int ul = (int) (u / c);
            int vl = (int) (v / c);
            byte cu = (byte) (u - ul * c);
            byte cv = (byte) (v - vl * c);

            ok.set(cu, cv);
            ov.set(ul, vl);

            context.write(ok, ov);

        }
    }

    public static class GPMapperSeq extends Mapper<LongWritable, LongWritable, BytePairWritable, IntPairWritable>{

        // the number of node colors
        int c;

        /**
         * Setup before execution
         * @param context
         */
        @Override
        protected void setup(Context context) {
            this.c = context.getConfiguration().getInt("numColors", 0);
        }

        BytePairWritable ok = new BytePairWritable();
        IntPairWritable ov = new IntPairWritable();

        /**
         * Parse an edge and emit the edge by its color.
         * @param key the first node of an edge
         * @param value the second node of an edge
         * @param context of Hadoop
         * @throws IOException of hadoop
         * @throws InterruptedException of Hadoop
         */
        @Override
        protected void map(LongWritable key, LongWritable value, Context context) throws IOException, InterruptedException {

            long u = key.get();
            long v = value.get();
            int ul = (int) (u / c);
            int vl = (int) (v / c);
            byte cu = (byte) (u - ul * c);
            byte cv = (byte) (v - vl * c);

            ok.set(cu, cv);
            ov.set(ul, vl);

            context.write(ok, ov);

        }
    }

    public static class GPPartitioner extends Partitioner<BytePairWritable, IntPairWritable>
            implements Configurable {

        int[][] precomputed_part_num;

        /**
         * get the partition of an entry
         * @param key of an entry
         * @param value of an entry
         * @param i the number of partitions
         * @return
         */
        @Override
        public int getPartition(BytePairWritable key, IntPairWritable value, int i) {
            return precomputed_part_num[key.get_u()][key.get_v()];
        }

        /**
         * compute partitions of all possible keys in advance
         * @param conf of Hadoop
         */
        @Override
        public void setConf(Configuration conf) {


            int c = conf.getInt("numColors", 0);
            int p = conf.getInt("mapred.reduce.tasks", 0);

            precomputed_part_num = new int[c][c];

            int i = 0;
            for (int u = 0; u < c; u++) {
                for (int v = 0; v < c; v++) {
                    precomputed_part_num[u][v] = i++ % p;
                }
            }

        }

        /**
         * not used
         * @return null
         */
        @Override
        public Configuration getConf() {
            return null;
        }
    }

    public static class GPReducer extends Reducer<BytePairWritable, IntPairWritable,
            NullWritable, NullWritable>{

        Path outputPath;
        Class<? extends Graph> gClass = null;

        /**
         * set the graph format
         * @param context of Hadoop
         */
        @Override
        protected void setup(Context context) {

            Configuration conf = context.getConfiguration();
            outputPath = new Path(conf.get("outputPath"));

            switch(conf.get("graphFormat", "csrv")){
                case "csrv":
                    gClass = CSRV.class;
                    break;
                case "csr":
                    gClass = CSR.class;
                    break;
            }
        }

        /**
         * write the graph
         * @param key the color of the graph
         * @param values edge list in the graph
         * @param context of Hadoop
         * @throws IOException by Hadoop
         */
        @Override
        protected void reduce(BytePairWritable key, Iterable<IntPairWritable> values, Context context) throws IOException {

            String ce_string = key.get_u() + "-" + key.get_v();

            Path edgepath = outputPath.suffix("/graph-" + ce_string + ".edge");
            Path nodepath = outputPath.suffix("/graph-" + ce_string + ".node");

            FileSystem fs = FileSystem.get(new Configuration());

            Output oedge = new Output(fs.create(edgepath));
            Output onode = new Output(fs.create(nodepath));

            try {
                gClass.newInstance().writeFrom(values, oedge, onode);
            } catch (Exception e) {
                e.printStackTrace();
            }

        }
    }

}
