package gasp.greedy;

import gasp.ep.EPManager;
import gasp.ep.ExtremePoint;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;

import java.util.List;

/**
 * Greedy constructor equivalent to EP-KPH.
 * Maintains the current packing and EP lists.
 */
public class GreedyState {

    private final Knapsack knapsack;
    private final MeritCriterion criterion;
    private final boolean allowRotation;
    
    private final Packing packing;
    private final EPManager epManager;
    
    private int envW = 0;
    private int envD = 0;
    
    private int itemsProcessed = 0;

    public GreedyState(Knapsack knapsack, MeritCriterion criterion, boolean allowRotation) {
        this.knapsack = knapsack;
        this.criterion = criterion;
        this.allowRotation = allowRotation;
        
        this.packing = new Packing(knapsack);
        this.epManager = new EPManager(knapsack);
    }

    public Packing getPacking() {
        return packing;
    }
    
    public EPManager getEpManager() {
        return epManager;
    }

    public int getItemsProcessed() {
        return itemsProcessed;
    }

    /**
     * Tries to place an item into the knapsack using the greedy criteria.
     *
     * @param item The item to place.
     * @return true if the item was successfully placed, false if it doesn't fit.
     */
    public boolean place(Item item) {
        List<int[]> rotations = item.rotations(allowRotation, knapsack.is3D());
        
        int[] bestRot = null;
        ExtremePoint bestEp = null;
        MeritValue bestMerit = null;
        
        List<ExtremePoint> eps = epManager.getEps();
        int epOrder = 0;
        
        for (int[] rot : rotations) {
            int w = rot[0];
            int d = rot[1];
            int h = rot[2];
            
            epOrder = 0;
            for (ExtremePoint ep : eps) {
                // Quick bounds check
                if (ep.getX() + w > knapsack.W() || 
                    ep.getY() + d > knapsack.D() || 
                    ep.getZ() + h > knapsack.H()) {
                    epOrder++;
                    continue;
                }
                
                Placement candidate = new Placement(item, ep.getX(), ep.getY(), ep.getZ(), w, d, h);
                if (!packing.feasible(candidate)) {
                    epOrder++;
                    continue;
                }
                
                // Feasible! Evaluate merit
                MeritValue merit = criterion.evaluate(ep, w, d, h, packing, envW, envD, epOrder);
                
                if (bestMerit == null || merit.compareTo(bestMerit) < 0) {
                    bestMerit = merit;
                    bestRot = rot;
                    bestEp = ep;
                }
                
                if (criterion == MeritCriterion.FF) {
                    break; // First Fit stops at the first valid EP for this rotation
                }
                
                epOrder++;
            }
            
            if (criterion == MeritCriterion.FF && bestEp != null) {
                break; // First Fit stops immediately after finding ANY valid placement
            }
        }
        
        itemsProcessed++;
        
        if (bestEp == null) {
            return false; // Could not place the item
        }
        
        // Apply placement
        Placement placed = new Placement(item, bestEp.getX(), bestEp.getY(), bestEp.getZ(), 
                                         bestRot[0], bestRot[1], bestRot[2]);
        packing.addPlacement(placed);
        
        // Update envelope for MP/LEV criteria
        if (bestEp.getX() + bestRot[0] > envW) envW = bestEp.getX() + bestRot[0];
        if (bestEp.getY() + bestRot[1] > envD) envD = bestEp.getY() + bestRot[1];
        
        // Update Extreme Points
        epManager.addItem(placed, packing);
        
        return true;
    }

    /**
     * Helper to run the greedy algorithm over a sequence of items.
     * @param items List of items to pack.
     * @return the resulting packing.
     */
    public Packing run(List<Item> items) {
        for (Item item : items) {
            place(item);
        }
        return packing;
    }
}
