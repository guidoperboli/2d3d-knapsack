package gasp.greedy;

import gasp.ems.Space;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ParrenoConstructTest {

    @Test
    public void testNearCorner() {
        Knapsack ks = new Knapsack(10, 10, 10);
        // Space closer to (0,0,0)
        Space s1 = new Space(1, 1, 1, 4, 4, 4);
        ParrenoConstruct.CornerResult cr1 = ParrenoConstruct.nearCorner(s1, ks);
        assertArrayEquals(new int[]{1, 1, 1}, cr1.dist);
        assertArrayEquals(new int[]{0, 0, 0}, cr1.sig);

        // Space closer to (10,10,10)
        Space s2 = new Space(7, 7, 7, 9, 9, 9);
        ParrenoConstruct.CornerResult cr2 = ParrenoConstruct.nearCorner(s2, ks);
        assertArrayEquals(new int[]{1, 1, 1}, cr2.dist);
        assertArrayEquals(new int[]{1, 1, 1}, cr2.sig);
    }

    @Test
    public void testPlaceBlock() {
        Space s = new Space(0, 0, 0, 10, 10, 10);
        int[] sig = {0, 0, 0};
        Item it1 = new Item(1, 2, 2, 2, 10.0);
        Item it2 = new Item(2, 2, 2, 2, 10.0);
        List<Item> members = Arrays.asList(it1, it2);
        
        ParrenoConstruct.BlockResult br = ParrenoConstruct.placeBlock(s, sig, 2, 2, 2, 2, 1, 1, members, 2);
        
        assertEquals(2, br.placements.size());
        assertEquals(0, br.box[0]);
        assertEquals(0, br.box[1]);
        assertEquals(0, br.box[2]);
        assertEquals(4, br.box[3]);
        assertEquals(2, br.box[4]);
        assertEquals(2, br.box[5]);
        
        assertEquals(0, br.placements.get(0).x());
        assertEquals(2, br.placements.get(1).x());
    }

    @Test
    public void testParrenoConstruct() {
        Knapsack ks = new Knapsack(10, 10, 10);
        List<Item> items = Arrays.asList(
                new Item(1, 5, 5, 5, 100),
                new Item(2, 5, 5, 5, 100),
                new Item(3, 5, 5, 5, 100),
                new Item(4, 5, 5, 5, 100),
                new Item(5, 5, 5, 5, 100),
                new Item(6, 5, 5, 5, 100),
                new Item(7, 5, 5, 5, 100),
                new Item(8, 5, 5, 5, 100)
        );

        Packing pk = ParrenoConstruct.parrenoConstruct(items, ks, true, "bestvol");
        
        // 8 items of 5x5x5 fit perfectly in a 10x10x10 container
        assertEquals(8, pk.getPlacements().size());
        assertEquals(1000, pk.usedVolume());
        assertEquals(800, pk.profit());
    }
}
