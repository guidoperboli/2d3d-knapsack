package gasp.alns;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ALNSTest {

    @Test
    public void testALNSExecution() {
        Knapsack ks = new Knapsack(10, 10, 10);
        List<Item> items = Arrays.asList(
                new Item(1, 5, 5, 5, 100),
                new Item(2, 5, 5, 5, 100),
                new Item(3, 5, 5, 5, 100),
                new Item(4, 5, 5, 5, 100),
                new Item(5, 5, 5, 5, 100),
                new Item(6, 5, 5, 5, 100),
                new Item(7, 5, 5, 5, 100),
                new Item(8, 5, 5, 5, 100),
                new Item(9, 2, 2, 2, 10),
                new Item(10, 2, 2, 2, 10)
        );

        ALNSParams params = new ALNSParams(10, 5.0, 0.1, 0.4, 0.05, 0.995, 200, 2.0, 0.5, 100, 1.5, 1.2, 0.8, false, "profit");

        ALNS alns = new ALNS(items, ks, params, 42L);
        ALNSResult result = alns.run();

        assertTrue(result.bestPacking().usedVolume() > 0, "ALNS should pack at least some items");
        assertTrue(result.iterations() > 0, "ALNS should run at least one iteration");
    }
}
