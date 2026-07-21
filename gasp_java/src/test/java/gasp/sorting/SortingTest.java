package gasp.sorting;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class SortingTest {

    @Test
    public void testProfitHeight() {
        Knapsack k = new Knapsack(10, 10, 10);
        Item i1 = new Item(1, 5, 5, 5, 10);
        Item i2 = new Item(2, 5, 5, 6, 10);
        Item i3 = new Item(3, 5, 5, 5, 20); // Highest profit
        
        List<Item> items = List.of(i1, i2, i3);
        List<Item> sorted = SortingRules.profitHeight(items, k, 10);
        
        // i3 has highest profit (20) -> should be first
        assertEquals(3, sorted.get(0).idx());
        
        // i1 and i2 have same profit (10). i2 has higher h (6 vs 5) -> should be second
        assertEquals(2, sorted.get(1).idx());
        assertEquals(1, sorted.get(2).idx());
    }

    @Test
    public void testClusteredProfitHeight() {
        Knapsack k = new Knapsack(10, 10, 10);
        // Band of 10% on profit range [10, 100] -> width = 90 * 0.1 = 9
        // c(10) = 0, c(15) = 0
        Item i1 = new Item(1, 5, 5, 8, 10);
        Item i2 = new Item(2, 5, 5, 4, 15);
        Item i3 = new Item(3, 5, 5, 2, 100); 
        
        List<Item> items = List.of(i1, i2, i3);
        List<Item> sorted = SortingRules.clusteredProfitHeight(items, k, 10);
        
        // i3 is in the top cluster
        assertEquals(3, sorted.get(0).idx());
        // i1 and i2 are in the same cluster (0). i1 has higher h (8 vs 4) -> should tie-break and be before i2,
        // even though i2 has slightly more absolute profit! This is the core mechanic of clustered sorting.
        assertEquals(1, sorted.get(1).idx());
        assertEquals(2, sorted.get(2).idx());
    }
}
