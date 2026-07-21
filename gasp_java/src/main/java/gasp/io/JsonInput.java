package gasp.io;

import java.util.List;

public class JsonInput {
    public KnapsackData knapsack;
    public List<ItemData> items;
    public ParamsData params;

    public static class KnapsackData {
        public int w;
        public int d;
        public int h;
    }

    public static class ItemData {
        public int idx;
        public int w;
        public int d;
        public int h;
        public double profit;
    }

    public static class ParamsData {
        public double time_limit;
        public double alpha;
        public double beta;
        public int k_init;
        public int non_improving_limit;
        public List<Integer> pch_deltas;
        public int reinit_swaps;
        public boolean allow_rotation;
    }
}
