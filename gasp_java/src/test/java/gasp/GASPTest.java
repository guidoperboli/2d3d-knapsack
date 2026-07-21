package gasp;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class GASPTest {

    @Test
    public void testGASPRun() {
        Knapsack k = new Knapsack(10, 10, 10);
        Item i1 = new Item(1, 10, 10, 5, 10);
        Item i2 = new Item(2, 10, 10, 5, 10);
        Item i3 = new Item(3, 10, 10, 5, 10); // 3 items of volume 500. Knapsack is 1000.
        
        List<Item> items = List.of(i1, i2, i3);
        
        // Short time limit for test
        GASPParams params = new GASPParams(0.5, 0.2, 0.1, 2, 10, new int[]{10}, 3, false, "classic", 0.05, 1.0, 0.99, false, false, false, "off");
        GASP gasp = new GASP(items, k, params, 42L);
        
        GASPResult result = gasp.run();
        
        assertNotNull(result);
        assertNotNull(result.bestPacking());
        
        // It should be able to pack exactly 2 items
        assertEquals(2, result.bestPacking().getPlacements().size());
        assertEquals(20.0, result.bestProfit());
        assertTrue(result.iterations() > 0);
        assertFalse(result.history().isEmpty());
    }
}
