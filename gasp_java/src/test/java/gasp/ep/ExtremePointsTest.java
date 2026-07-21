package gasp.ep;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;
import org.junit.jupiter.api.Test;

import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class ExtremePointsTest {

    @Test
    public void testEPManagerInitialization() {
        Knapsack ks = new Knapsack(100, 100, 100);
        EPManager manager = new EPManager(ks);
        
        List<ExtremePoint> eps = manager.getEps();
        assertEquals(1, eps.size());
        
        ExtremePoint origin = eps.get(0);
        assertEquals(0, origin.getX());
        assertEquals(0, origin.getY());
        assertEquals(0, origin.getZ());
        
        assertEquals(100, origin.getRsX());
        assertEquals(100, origin.getRsY());
        assertEquals(100, origin.getRsZ());
    }

    @Test
    public void testEPManagerAddItem() {
        Knapsack ks = new Knapsack(100, 100, 100);
        Packing packing = new Packing(ks);
        EPManager manager = new EPManager(ks);
        
        // Add first item at origin
        Item item1 = new Item(1, 10, 20, 30, 1);
        Placement p1 = new Placement(item1, 0, 0, 0, 10, 20, 30);
        packing.addPlacement(p1);
        manager.addItem(p1, packing);
        
        List<ExtremePoint> eps = manager.getEps();
        
        // The origin should be removed as it's covered by the item.
        // EPs generated from (0,0,0) with sizes (10,20,30) in 3D:
        // (10, 0, 0)
        // (0, 20, 0)
        // (0, 0, 30)
        // Since there are no other items, these should be the only generated EPs that are valid.
        // Wait, the projection logic generates multiple EPs.
        assertFalse(eps.contains(new ExtremePoint(0,0,0, 0,0,0)));
        
        boolean foundX = false;
        boolean foundY = false;
        boolean foundZ = false;
        
        for (ExtremePoint ep : eps) {
            if (ep.getX() == 10 && ep.getY() == 0 && ep.getZ() == 0) {
                foundX = true;
                assertEquals(90, ep.getRsX()); // 100 - 10
                assertEquals(100, ep.getRsY());
                assertEquals(100, ep.getRsZ());
            }
            if (ep.getX() == 0 && ep.getY() == 20 && ep.getZ() == 0) {
                foundY = true;
                assertEquals(100, ep.getRsX());
                assertEquals(80, ep.getRsY()); // 100 - 20
                assertEquals(100, ep.getRsZ());
            }
            if (ep.getX() == 0 && ep.getY() == 0 && ep.getZ() == 30) {
                foundZ = true;
                assertEquals(100, ep.getRsX());
                assertEquals(100, ep.getRsY());
                assertEquals(70, ep.getRsZ()); // 100 - 30
            }
        }
        
        assertTrue(foundX, "EP at (10,0,0) not found");
        assertTrue(foundY, "EP at (0,20,0) not found");
        assertTrue(foundZ, "EP at (0,0,30) not found");
    }
}
