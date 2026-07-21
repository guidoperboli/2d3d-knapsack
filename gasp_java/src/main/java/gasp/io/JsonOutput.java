package gasp.io;

import java.util.List;

public class JsonOutput {
    public double profit;
    public long volume;
    public int iterations;
    public double elapsed;
    public List<PlacementData> placements;

    public static class PlacementData {
        public int idx;
        public int x;
        public int y;
        public int z;
        public int w;
        public int d;
        public int h;
    }
}
