package gasp.greedy;

import gasp.ep.ExtremePoint;
import gasp.geometry.Packing;
import gasp.geometry.Placement;

import java.util.List;

/**
 * EP evaluation criteria (merit functions) of Section 3.1.
 * Lower merit values are better.
 */
public enum MeritCriterion {

    /** First Fit: the first compatible EP is taken. */
    FF {
        @Override
        public MeritValue evaluate(ExtremePoint ep, int w, int d, int h, Packing packing, int envW, int envD, int order) {
            // The first compatible EP wins: merit is the discovery order
            return new MeritValue(order, 0, 0, 0);
        }
    },

    /** Minimize the maximum packing size on the X and Y axes. */
    MP {
        @Override
        public MeritValue evaluate(ExtremePoint ep, int w, int d, int h, Packing packing, int envW, int envD, int order) {
            long fx = (ep.getX() + w > envW) ? (ep.getX() + w - envW) : 0;
            long fy = (ep.getY() + d > envD) ? (ep.getY() + d - envD) : 0;
            return new MeritValue(fx + fy, ep.getZ(), ep.getY(), ep.getX());
        }
    },

    /** Level the packing on the X and Y axes. */
    LEV {
        @Override
        public MeritValue evaluate(ExtremePoint ep, int w, int d, int h, Packing packing, int envW, int envD, int order) {
            long C = Math.max(packing.getKnapsack().W(), packing.getKnapsack().D()) + 1L;
            long fx = (ep.getX() + w > envW) ? (ep.getX() + w - envW) * C : (envW - (ep.getX() + w));
            long fy = (ep.getY() + d > envD) ? (ep.getY() + d - envD) * C : (envD - (ep.getY() + d));
            return new MeritValue(fx + fy, ep.getZ(), ep.getY(), ep.getX());
        }
    },

    /** Maximize the utilization of the EPs' Residual Space. */
    RS {
        @Override
        public MeritValue evaluate(ExtremePoint ep, int w, int d, int h, Packing packing, int envW, int envD, int order) {
            long f = (ep.getRsX() - w) + (ep.getRsY() - d) + (ep.getRsZ() - h);
            return new MeritValue(f, ep.getZ(), ep.getY(), ep.getX());
        }
    },

    /** Touching Perimeter: maximize the surface shared with knapsack walls and items. */
    TP {
        @Override
        public MeritValue evaluate(ExtremePoint ep, int w, int d, int h, Packing packing, int envW, int envD, int order) {
            long contact = contactArea(ep.getX(), ep.getY(), ep.getZ(), w, d, h, packing);
            // We want to maximize contact, but lower merit is better, so negate it
            return new MeritValue(-contact, ep.getZ(), ep.getY(), ep.getX());
        }
    };

    /**
     * Evaluates a candidate placement.
     *
     * @param ep      The Extreme Point
     * @param w       Width of the item rotation
     * @param d       Depth of the item rotation
     * @param h       Height of the item rotation
     * @param packing The current packing state
     * @param envW    Current envelope X size
     * @param envD    Current envelope Y size
     * @param order   The discovery order of the EP (used by FF)
     * @return A MeritValue tuple for comparison
     */
    public abstract MeritValue evaluate(ExtremePoint ep, int w, int d, int h, Packing packing, int envW, int envD, int order);

    // --- Helper methods for TP (Touching Perimeter) ---

    private static long overlap1D(int a1, int a2, int b1, int b2) {
        int lo = Math.max(a1, b1);
        int hi = Math.min(a2, b2);
        return hi > lo ? hi - lo : 0;
    }

    private static long contactArea(int x, int y, int z, int w, int d, int h, Packing packing) {
        int W = packing.getKnapsack().W();
        int D = packing.getKnapsack().D();
        int H = packing.getKnapsack().H();
        long c = 0;

        if (x == 0) c += (long) d * h;
        if (x + w == W) c += (long) d * h;
        if (y == 0) c += (long) w * h;
        if (y + d == D) c += (long) w * h;
        if (z == 0) c += (long) w * d;
        if (z + h == H) c += (long) w * d;

        List<Placement> placements = packing.getPlacements();
        for (Placement p : placements) {
            if (x + w == p.x() || p.x2() == x) {
                c += overlap1D(y, y + d, p.y(), p.y2()) * overlap1D(z, z + h, p.z(), p.z2());
            }
            if (y + d == p.y() || p.y2() == y) {
                c += overlap1D(x, x + w, p.x(), p.x2()) * overlap1D(z, z + h, p.z(), p.z2());
            }
            if (z + h == p.z() || p.z2() == z) {
                c += overlap1D(x, x + w, p.x(), p.x2()) * overlap1D(y, y + d, p.y(), p.y2());
            }
        }
        return c;
    }
}
