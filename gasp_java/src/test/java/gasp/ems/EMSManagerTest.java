package gasp.ems;

import org.junit.jupiter.api.Test;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class EMSManagerTest {

    @Test
    public void testDifferenceProcess() {
        List<Space> spaces = new ArrayList<>();
        spaces.add(new Space(0, 0, 0, 10, 10, 10));

        // Place a box at origin 2x2x2
        List<Space> newSpaces = EMSManager.differenceProcess(spaces, 0, 0, 0, 2, 2, 2, 1);

        // We should have 3 spaces left (since the others are dominated)
        // Space 1: x from 2 to 10, y:0-10, z:0-10
        // Space 2: y from 2 to 10, x:0-10, z:0-10
        // Space 3: z from 2 to 10, x:0-10, y:0-10
        assertEquals(3, newSpaces.size(), "Should have 3 non-dominated spaces");

        boolean hasXSpace = false;
        boolean hasYSpace = false;
        boolean hasZSpace = false;

        for (Space s : newSpaces) {
            if (s.x == 2 && s.y == 0 && s.z == 0 && s.x2 == 10 && s.y2 == 10 && s.z2 == 10) hasXSpace = true;
            if (s.x == 0 && s.y == 2 && s.z == 0 && s.x2 == 10 && s.y2 == 10 && s.z2 == 10) hasYSpace = true;
            if (s.x == 0 && s.y == 0 && s.z == 2 && s.x2 == 10 && s.y2 == 10 && s.z2 == 10) hasZSpace = true;
        }

        assertTrue(hasXSpace, "Missing space along X");
        assertTrue(hasYSpace, "Missing space along Y");
        assertTrue(hasZSpace, "Missing space along Z");
    }

    @Test
    public void testRemoveDominated() {
        List<Space> spaces = new ArrayList<>();
        spaces.add(new Space(0, 0, 0, 10, 10, 10)); // Largest
        spaces.add(new Space(0, 0, 0, 5, 5, 5));    // Dominated by the first
        spaces.add(new Space(10, 10, 10, 15, 15, 15)); // Not dominated

        List<Space> result = EMSManager.removeDominated(spaces, 1);
        assertEquals(2, result.size());
        
        // Due to descending volume sort
        assertEquals(1000, result.get(0).getVolume());
        assertEquals(125, result.get(1).getVolume());
    }
}
