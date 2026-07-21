package gasp.geometry;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * A (partial) solution: the set of placements inside the knapsack.
 */
public class Packing {

    private final Knapsack knapsack;
    private final List<Placement> placements;

    public Packing(Knapsack knapsack) {
        this.knapsack = knapsack;
        this.placements = new ArrayList<>();
    }

    public Packing(Knapsack knapsack, List<Placement> placements) {
        this.knapsack = knapsack;
        this.placements = new ArrayList<>(placements);
    }

    public Knapsack getKnapsack() {
        return knapsack;
    }

    public List<Placement> getPlacements() {
        return placements;
    }

    public void addPlacement(Placement placement) {
        this.placements.add(placement);
    }

    public double profit() {
        double totalProfit = 0.0;
        for (Placement p : placements) {
            totalProfit += p.item().profit();
        }
        return totalProfit;
    }

    public int usedVolume() {
        int totalVolume = 0;
        for (Placement p : placements) {
            totalVolume += p.w() * p.d() * p.h();
        }
        return totalVolume;
    }

    public Set<Integer> loadedIds() {
        Set<Integer> ids = new HashSet<>();
        for (Placement p : placements) {
            ids.add(p.item().idx());
        }
        return ids;
    }

    /**
     * Minimum box envelope of the current packing.
     * @return an array [x2, y2, z2] representing the envelope dimensions.
     */
    public int[] envelope() {
        if (placements.isEmpty()) {
            return new int[]{0, 0, 0};
        }
        int maxX = 0;
        int maxY = 0;
        int maxZ = 0;
        for (Placement p : placements) {
            if (p.x2() > maxX) maxX = p.x2();
            if (p.y2() > maxY) maxY = p.y2();
            if (p.z2() > maxZ) maxZ = p.z2();
        }
        return new int[]{maxX, maxY, maxZ};
    }

    /**
     * Checks if a candidate placement is feasible within the current packing.
     *
     * @param cand the candidate placement
     * @return true if feasible, false otherwise
     */
    public boolean feasible(Placement cand) {
        if (!knapsack.fits(cand.x(), cand.y(), cand.z(), cand.w(), cand.d(), cand.h())) {
            return false;
        }
        // Check for overlaps using a simple loop for performance (avoids stream overhead)
        for (Placement p : placements) {
            if (cand.overlaps(p)) {
                return false;
            }
        }
        return true;
    }
}
