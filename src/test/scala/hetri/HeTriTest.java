package hetri;

import hetri.gp.GraphPartitioner;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.junit.Test;

import java.net.URL;

import static org.junit.Assert.assertEquals;

public class HeTriTest {

    @Test
    public void testAll() throws Exception {

        URL[] paths = {
                getClass().getResource("/graph/facebook"),
        };

        int numColors = 10;


        FileSystem fs = FileSystem.get(new Configuration());

        for (URL path : paths) {
            String input = path.getPath();
            String part = input + ".part";
            Path partPath = new Path(part);

            Configuration conf = new Configuration();
            conf.setInt("numColors", numColors);
            conf.set("graphFormat", "csrv");

            GraphPartitioner gp = new GraphPartitioner();
            ToolRunner.run(conf, gp, new String[]{input, part});

            HeTri hetri = new HeTri();

            ToolRunner.run(conf, hetri, new String[]{input});

            assertEquals(1612010, hetri.numTriangles());
        }

    }

    private Path hdfsBasePath(int eid, Path part) {
        String ceString = ((eid >> 8) & 0xFF) + "-" + (eid & 0xFF);
        return part.suffix("/graph-" + ceString);
    }

}
