package gasp;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;
import gasp.io.JsonInput;
import gasp.io.JsonOutput;
import gasp.alns.ALNS;
import gasp.alns.ALNSParams;
import gasp.alns.ALNSResult;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);

        JsonInput input;
        if (args.length > 0) {
            input = mapper.readValue(new File(args[0]), JsonInput.class);
        } else {
            input = mapper.readValue(System.in, JsonInput.class);
        }

        // Convert Input DTO to domain objects
        Knapsack knapsack = new Knapsack(input.knapsack.w, input.knapsack.d, input.knapsack.h);
        
        List<Item> items = new ArrayList<>();
        for (JsonInput.ItemData itemData : input.items) {
            items.add(new Item(itemData.idx, itemData.w, itemData.d, itemData.h, itemData.profit));
        }

        // Parse params or use defaults
        GASPParams params = GASPParams.defaultParams();
        if (input.params != null) {
            int[] deltas = {10};
            if (input.params.pch_deltas != null && !input.params.pch_deltas.isEmpty()) {
                deltas = input.params.pch_deltas.stream().mapToInt(Integer::intValue).toArray();
            }
            params = new GASPParams(
                input.params.time_limit > 0 ? input.params.time_limit : params.timeLimit(),
                input.params.alpha > 0 ? input.params.alpha : params.alpha(),
                input.params.beta > 0 ? input.params.beta : params.beta(),
                input.params.k_init > 0 ? input.params.k_init : params.kInit(),
                input.params.non_improving_limit > 0 ? input.params.non_improving_limit : params.nonImprovingLimit(),
                deltas,
                input.params.reinit_swaps > 0 ? input.params.reinit_swaps : params.reinitSwaps(),
                input.params.allow_rotation,
                
                input.params.update_policy != null ? input.params.update_policy : params.updatePolicy(),
                input.params.band_fraction > 0 ? input.params.band_fraction : params.bandFraction(),
                input.params.policy_reward > 0 ? input.params.policy_reward : params.policyReward(),
                input.params.policy_decay > 0 ? input.params.policy_decay : params.policyDecay(),
                
                input.params.layer_greedy,
                input.params.use_ems,
                input.params.parreno_seed,
                input.params.block_mode != null ? input.params.block_mode : params.blockMode()
            );
        }

        // Run GASP or ALNS based on solver flag
        String solver = input.solver != null ? input.solver.toLowerCase() : "gasp";
        
        Packing bestPacking;
        double bestProfit;
        int iterations;
        double elapsed;
        
        if ("alns".equals(solver)) {
            ALNSParams alnsParams = ALNSParams.defaultParams();
            if (input.params != null) {
                alnsParams = new ALNSParams(
                    input.params.max_iter > 0 ? input.params.max_iter : alnsParams.maxIter(),
                    input.params.time_limit > 0 ? input.params.time_limit : alnsParams.timeLimit(),
                    input.params.frac_lo > 0 ? input.params.frac_lo : alnsParams.fracLo(),
                    input.params.frac_hi > 0 ? input.params.frac_hi : alnsParams.fracHi(),
                    input.params.T0_ratio > 0 ? input.params.T0_ratio : alnsParams.t0Ratio(),
                    input.params.cooling > 0 ? input.params.cooling : alnsParams.cooling(),
                    input.params.reheat_after > 0 ? input.params.reheat_after : alnsParams.reheatAfter(),
                    input.params.reheat_ratio > 0 ? input.params.reheat_ratio : alnsParams.reheatRatio(),
                    input.params.react > 0 ? input.params.react : alnsParams.react(),
                    input.params.seg_update > 0 ? input.params.seg_update : alnsParams.segUpdate(),
                    input.params.reward_best > 0 ? input.params.reward_best : alnsParams.rewardBest(),
                    input.params.reward_better > 0 ? input.params.reward_better : alnsParams.rewardBetter(),
                    input.params.reward_accept > 0 ? input.params.reward_accept : alnsParams.rewardAccept(),
                    input.params.allow_rotation,
                    input.params.objective_metric != null ? input.params.objective_metric : alnsParams.objectiveMetric()
                );
            }
            ALNS alns = new ALNS(items, knapsack, alnsParams, null);
            ALNSResult result = alns.run();
            bestPacking = result.bestPacking();
            bestProfit = result.bestProfit();
            iterations = result.iterations();
            elapsed = result.elapsedSeconds();
        } else {
            GASP gasp = new GASP(items, knapsack, params, null);
            GASPResult result = gasp.run();
            bestPacking = result.bestPacking();
            bestProfit = result.bestProfit();
            iterations = result.iterations();
            elapsed = result.elapsedSeconds();
        }

        // Convert domain objects to Output DTO
        JsonOutput output = new JsonOutput();
        output.profit = bestProfit;
        output.volume = bestPacking.usedVolume();
        output.iterations = iterations;
        output.elapsed = elapsed;
        
        output.placements = new ArrayList<>();
        for (Placement p : bestPacking.getPlacements()) {
            JsonOutput.PlacementData pd = new JsonOutput.PlacementData();
            pd.idx = p.item().idx();
            pd.x = p.x();
            pd.y = p.y();
            pd.z = p.z();
            pd.w = p.w();
            pd.d = p.d();
            pd.h = p.h();
            output.placements.add(pd);
        }

        // Write output
        mapper.writeValue(System.out, output);
    }
}
