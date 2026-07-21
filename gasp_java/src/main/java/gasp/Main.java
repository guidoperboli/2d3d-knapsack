package gasp;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Placement;
import gasp.io.JsonInput;
import gasp.io.JsonOutput;

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
                input.params.allow_rotation
            );
        }

        // Run GASP
        GASP gasp = new GASP(items, knapsack, params, null);
        GASPResult result = gasp.run();

        // Convert domain objects to Output DTO
        JsonOutput output = new JsonOutput();
        output.profit = result.bestProfit();
        output.volume = result.bestPacking().usedVolume();
        output.iterations = result.iterations();
        output.elapsed = result.elapsedSeconds();
        
        output.placements = new ArrayList<>();
        for (Placement p : result.bestPacking().getPlacements()) {
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
