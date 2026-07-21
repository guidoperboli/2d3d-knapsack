package gasp.ep;

import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Maintains the EP list and the residual spaces of a packing.
 */
public class EPManager {

    private final Knapsack knapsack;
    private List<ExtremePoint> eps;

    // By default extended projections are off, matching Python.
    private static final boolean EXTENDED_PROJECTIONS = false;

    public EPManager(Knapsack knapsack) {
        this.knapsack = knapsack;
        this.eps = new ArrayList<>();
        // The origin is the first EP, with RS equal to the knapsack itself
        this.eps.add(new ExtremePoint(0, 0, 0, knapsack.W(), knapsack.D(), knapsack.H()));
    }

    public List<ExtremePoint> getEps() {
        return eps;
    }

    private int projectX(int x, int y, int z, List<Placement> placements) {
        int best = 0;
        for (Placement p : placements) {
            if (p.x2() <= x && p.y() <= y && y < p.y2() && p.z() <= z && z < p.z2()) {
                if (p.x2() > best) best = p.x2();
            }
        }
        return best;
    }

    private int projectY(int x, int y, int z, List<Placement> placements) {
        int best = 0;
        for (Placement p : placements) {
            if (p.y2() <= y && p.x() <= x && x < p.x2() && p.z() <= z && z < p.z2()) {
                if (p.y2() > best) best = p.y2();
            }
        }
        return best;
    }

    private int projectZ(int x, int y, int z, List<Placement> placements) {
        int best = 0;
        for (Placement p : placements) {
            if (p.z2() <= z && p.x() <= x && x < p.x2() && p.y() <= y && y < p.y2()) {
                if (p.z2() > best) best = p.z2();
            }
        }
        return best;
    }

    private ExtremePoint residualSpace(int x, int y, int z, List<Placement> placements) {
        int rsX = knapsack.W() - x;
        int rsY = knapsack.D() - y;
        int rsZ = knapsack.H() - z;
        for (Placement p : placements) {
            // item ahead on X, overlapping on Y and Z
            if (p.x() >= x && p.y() < y + 1 && y + 1 <= p.y2() && p.z() < z + 1 && z + 1 <= p.z2()) {
                rsX = Math.min(rsX, p.x() - x);
            }
            if (p.y() >= y && p.x() < x + 1 && x + 1 <= p.x2() && p.z() < z + 1 && z + 1 <= p.z2()) {
                rsY = Math.min(rsY, p.y() - y);
            }
            if (p.z() >= z && p.x() < x + 1 && x + 1 <= p.x2() && p.y() < y + 1 && y + 1 <= p.y2()) {
                rsZ = Math.min(rsZ, p.z() - z);
            }
        }
        return new ExtremePoint(x, y, z, rsX, rsY, rsZ);
    }

    /**
     * Update the EP list after `placed` has been added to `packing`.
     * The placement is assumed to be already appended to packing.placements.
     */
    public void addItem(Placement placed, Packing packing) {
        List<Placement> allPlacements = packing.getPlacements();
        List<Placement> others = new ArrayList<>(allPlacements.size() - 1);
        for (Placement p : allPlacements) {
            if (p != placed) {
                others.add(p);
            }
        }

        int x = placed.x();
        int y = placed.y();
        int z = placed.z();
        int w = placed.w();
        int d = placed.d();
        int h = placed.h();

        List<int[]> newPts = new ArrayList<>();

        // Projections of (x + w, y, z) on Y and Z directions
        newPts.add(new int[]{x + w, projectY(x + w, y, z, others), z});
        newPts.add(new int[]{x + w, y, projectZ(x + w, y, z, others)});
        // Projections of (x, y + d, z) on X and Z directions
        newPts.add(new int[]{projectX(x, y + d, z, others), y + d, z});
        newPts.add(new int[]{x, y + d, projectZ(x, y + d, z, others)});

        if (knapsack.is3D()) {
            // Projections of (x, y, z + h) on X and Y directions
            newPts.add(new int[]{projectX(x, y, z + h, others), y, z + h});
            newPts.add(new int[]{x, projectY(x, y, z + h, others), z + h});

            // Composed projections
            int py1 = newPts.get(0)[1];
            newPts.add(new int[]{x + w, py1, projectZ(x + w, py1, z, others)});
            int pz1 = newPts.get(1)[2];
            newPts.add(new int[]{x + w, projectY(x + w, y, pz1, others), pz1});
            int px1 = newPts.get(2)[0];
            newPts.add(new int[]{px1, y + d, projectZ(px1, y + d, z, others)});
            int pz2 = newPts.get(3)[2];
            newPts.add(new int[]{projectX(x, y + d, pz2, others), y + d, pz2});
            int px2 = newPts.get(4)[0];
            newPts.add(new int[]{px2, projectY(px2, y, z + h, others), z + h});
            int py2 = newPts.get(5)[1];
            newPts.add(new int[]{projectX(x, py2, z + h, others), py2, z + h});
            
            // Note: EXTENDED_PROJECTIONS logic could be added here if needed,
            // skipped for brevity as it is default off in Python.
        }

        // Remove EPs covered by the new item, keep the others
        List<ExtremePoint> survivors = new ArrayList<>();
        for (ExtremePoint ep : this.eps) {
            boolean inside = (placed.x() <= ep.getX() && ep.getX() < placed.x2()
                    && placed.y() <= ep.getY() && ep.getY() < placed.y2()
                    && placed.z() <= ep.getZ() && ep.getZ() < placed.z2());
            if (!inside) {
                survivors.add(ep);
            }
        }
        this.eps = survivors;

        Set<String> existing = new HashSet<>();
        for (ExtremePoint ep : this.eps) {
            existing.add(ep.getX() + "," + ep.getY() + "," + ep.getZ());
        }

        for (int[] pt : newPts) {
            int px = pt[0], py = pt[1], pz = pt[2];
            String key = px + "," + py + "," + pz;
            if (existing.contains(key)) {
                continue;
            }
            if (px >= knapsack.W() || py >= knapsack.D() || pz >= knapsack.H()) {
                continue;
            }
            this.eps.add(residualSpace(px, py, pz, allPlacements)); // allPlacements includes placed
            existing.add(key);
        }

        // Update the RS of all the EPs against the new item
        for (ExtremePoint ep : this.eps) {
            if (placed.x() >= ep.getX() && placed.y() <= ep.getY() && ep.getY() < placed.y2()
                    && placed.z() <= ep.getZ() && ep.getZ() < placed.z2()) {
                ep.setRsX(Math.min(ep.getRsX(), placed.x() - ep.getX()));
            }
            if (placed.y() >= ep.getY() && placed.x() <= ep.getX() && ep.getX() < placed.x2()
                    && placed.z() <= ep.getZ() && ep.getZ() < placed.z2()) {
                ep.setRsY(Math.min(ep.getRsY(), placed.y() - ep.getY()));
            }
            if (placed.z() >= ep.getZ() && placed.x() <= ep.getX() && ep.getX() < placed.x2()
                    && placed.y() <= ep.getY() && ep.getY() < placed.y2()) {
                ep.setRsZ(Math.min(ep.getRsZ(), placed.z() - ep.getZ()));
            }
        }
    }
}
