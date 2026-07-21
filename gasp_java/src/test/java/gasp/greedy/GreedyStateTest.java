package gasp.greedy;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;
import org.junit.jupiter.api.Test;

import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class GreedyStateTest {

    @Test
    public void testGreedyRS() {
        Knapsack ks = new Knapsack(10, 10, 10);
        GreedyState state = new GreedyState(ks, MeritCriterion.RS, false); // No rotation
        
        // Item 1: 5x5x5
        Item item1 = new Item(1, 5, 5, 5, 10);
        assertTrue(state.place(item1));
        
        Packing packing = state.getPacking();
        assertEquals(1, packing.getPlacements().size());
        
        Placement p1 = packing.getPlacements().get(0);
        // RS minimizes the residual space distance, and origin (0,0,0) is always the first EP 
        // with maximum RS, so placing there maximizes utilization.
        assertEquals(0, p1.x());
        assertEquals(0, p1.y());
        assertEquals(0, p1.z());
        
        // Item 2: 5x5x5
        Item item2 = new Item(2, 5, 5, 5, 10);
        assertTrue(state.place(item2));
        
        assertEquals(2, packing.getPlacements().size());
        Placement p2 = packing.getPlacements().get(1);
        
        // Next best EP should be adjacent (e.g., 5,0,0 or 0,5,0 or 0,0,5)
        assertTrue((p2.x() == 5 && p2.y() == 0 && p2.z() == 0) ||
                   (p2.x() == 0 && p2.y() == 5 && p2.z() == 0) ||
                   (p2.x() == 0 && p2.y() == 0 && p2.z() == 5));
    }

    @Test
    public void testGreedyCannotFit() {
        Knapsack ks = new Knapsack(5, 5, 5);
        GreedyState state = new GreedyState(ks, MeritCriterion.RS, false);
        
        // Try to place a 6x6x6 item
        Item tooBig = new Item(1, 6, 6, 6, 10);
        assertFalse(state.place(tooBig));
        assertEquals(0, state.getPacking().getPlacements().size());
    }

    @Test
    public void testGreedyRunSequence() {
        Knapsack ks = new Knapsack(10, 10, 10);
        GreedyState state = new GreedyState(ks, MeritCriterion.MP, false);
        
        Item item1 = new Item(1, 5, 10, 10, 10);
        Item item2 = new Item(2, 5, 10, 10, 10);
        
        state.run(List.of(item1, item2));
        
        Packing packing = state.getPacking();
        assertEquals(2, packing.getPlacements().size());
        assertEquals(1000, packing.usedVolume());
        assertEquals(ks.volume(), packing.usedVolume());
    }
}
