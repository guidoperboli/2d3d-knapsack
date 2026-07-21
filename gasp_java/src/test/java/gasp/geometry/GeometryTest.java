package gasp.geometry;

import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class GeometryTest {

    @Test
    public void testItemRotations3D() {
        Item item = new Item(1, 10, 20, 30, 100); // 3D item
        List<int[]> rots = item.rotations(true, true);
        
        // 10,20,30 should produce 6 distinct rotations if allowRotation is true
        assertEquals(6, rots.size());
        
        // Check sorting
        assertArrayEquals(new int[]{10, 20, 30}, rots.get(0));
        assertArrayEquals(new int[]{10, 30, 20}, rots.get(1));
    }

    @Test
    public void testItemRotationsNoRotation() {
        Item item = new Item(1, 10, 20, 30, 100);
        List<int[]> rots = item.rotations(false, true);
        assertEquals(1, rots.size());
        assertArrayEquals(new int[]{10, 20, 30}, rots.get(0));
    }

    @Test
    public void testItemRotationsVFlags() {
        // vh = false, meaning height (30) cannot be placed vertically
        Item item = new Item(1, 10, 20, 30, 100, true, true, false, true);
        List<int[]> rots = item.rotations(true, true);
        
        // Allowed vertical values: 10, 20. Value 30 cannot be the 3rd element.
        // Out of 6 permutations, 2 have 30 as height. So 4 should remain.
        assertEquals(4, rots.size());
        for (int[] rot : rots) {
            assertNotEquals(30, rot[2]);
        }
    }

    @Test
    public void testPlacementOverlaps() {
        Item item1 = new Item(1, 10, 10, 10, 1);
        Placement p1 = new Placement(item1, 0, 0, 0, 10, 10, 10);
        
        Placement p2 = new Placement(item1, 5, 5, 5, 10, 10, 10); // Overlaps
        assertTrue(p1.overlaps(p2));
        assertTrue(p2.overlaps(p1));
        
        Placement p3 = new Placement(item1, 10, 0, 0, 10, 10, 10); // Touching, no overlap
        assertFalse(p1.overlaps(p3));
        assertFalse(p3.overlaps(p1));
    }

    @Test
    public void testKnapsackFits() {
        Knapsack ks = new Knapsack(100, 100, 100);
        assertTrue(ks.fits(90, 90, 90, 10, 10, 10)); // Exact fit
        assertFalse(ks.fits(90, 90, 90, 11, 10, 10)); // Exceeds W
    }

    @Test
    public void testPackingFeasible() {
        Knapsack ks = new Knapsack(100, 100, 100);
        Packing packing = new Packing(ks);
        
        Item item1 = new Item(1, 10, 10, 10, 1);
        Placement p1 = new Placement(item1, 0, 0, 0, 10, 10, 10);
        
        assertTrue(packing.feasible(p1));
        packing.addPlacement(p1);
        
        Placement p2 = new Placement(item1, 5, 5, 5, 10, 10, 10);
        assertFalse(packing.feasible(p2)); // Overlaps with p1
        
        Placement p3 = new Placement(item1, 10, 0, 0, 10, 10, 10);
        assertTrue(packing.feasible(p3)); // Next to p1
    }
}
