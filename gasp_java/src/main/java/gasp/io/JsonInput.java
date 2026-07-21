package gasp.io;

import java.util.List;

public class JsonInput {
    public KnapsackData knapsack;
    public List<ItemData> items;
    public ParamsData params;
    public String solver;

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
        
        // Adaptive Policies
        public String update_policy;
        public double band_fraction;
        public double policy_reward;
        public double policy_decay;
        
        // 3D Constructives
        public boolean layer_greedy;
        public boolean use_ems;
        public boolean parreno_seed;
        public String block_mode;
        
        // ALNS
        public int max_iter;
        public double frac_lo;
        public double frac_hi;
        public double T0_ratio;
        public double cooling;
        public int reheat_after;
        public double reheat_ratio;
        public double react;
        public int seg_update;
        public double reward_best;
        public double reward_better;
        public double reward_accept;
        public String objective_metric;
    }
}
