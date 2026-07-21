package gasp.greedy;

import gasp.ems.EMSManager;
import gasp.ems.Space;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;

import java.util.ArrayList;
import java.util.List;

public class EMSGreedy {
    private final Knapsack ks;
    private final String criterion; // "VOL" or "BSS"
    private final boolean allowRotation;

    public EMSGreedy(Knapsack ks, String criterion, boolean allowRotation) {
        this.ks = ks;
        this.criterion = criterion;
        this.allowRotation = allowRotation;
    }

    public Packing run(List<Item> itemsInOrder) {
        List<Space> spaces = new ArrayList<>();
        spaces.add(new Space(0, 0, 0, ks.W(), ks.D(), ks.H()));
        List<Placement> placements = new ArrayList<>();
        boolean bss = "BSS".equals(criterion);

        for (Item it : itemsInOrder) {
            int[] best = null; // {bx, by, bz, w, d, h}
            Object[] bestScore = null;

            for (Space s : spaces) {
                int fw = s.getW();
                int fd = s.getD();
                int fh = s.getH();

                for (int[] rot : it.rotations(allowRotation, ks.is3D())) {
                    int w = rot[0], d = rot[1], h = rot[2];
                    if (w > fw || d > fd || h > fh) continue;

                    Object[] score;
                    if (bss) {
                        int m0 = fw - w;
                        int m1 = fd - d;
                        int m2 = fh - h;
                        
                        // Sort m0, m1, m2
                        if (m1 < m0) { int t = m0; m0 = m1; m1 = t; }
                        if (m2 < m1) { int t = m1; m1 = m2; m2 = t; }
                        if (m1 < m0) { int t = m0; m0 = m1; m1 = t; }

                        score = new Object[]{m0, m1, m2, s.z, s.y, s.x};
                    } else {
                        long leftover = s.getVolume() - (long) w * d * h;
                        score = new Object[]{leftover, s.z, s.y, s.x};
                    }

                    if (bestScore == null || compareScores(score, bestScore, bss) < 0) {
                        bestScore = score;
                        best = new int[]{s.x, s.y, s.z, w, d, h};
                    }
                }
            }

            if (best == null) continue;

            int bx = best[0], by = best[1], bz = best[2];
            int w = best[3], d = best[4], h = best[5];

            placements.add(new Placement(it, bx, by, bz, w, d, h));
            int bx2 = bx + w, by2 = by + d, bz2 = bz + h;
            
            spaces = EMSManager.differenceProcess(spaces, bx, by, bz, bx2, by2, bz2, 1);
            if (spaces.isEmpty()) break;
        }

        return new Packing(ks, placements);
    }

    private int compareScores(Object[] a, Object[] b, boolean bss) {
        if (bss) {
            for (int i = 0; i < 3; i++) {
                int valA = (int) a[i];
                int valB = (int) b[i];
                if (valA != valB) return Integer.compare(valA, valB);
            }
            for (int i = 3; i < 6; i++) {
                int valA = (int) a[i];
                int valB = (int) b[i];
                if (valA != valB) return Integer.compare(valA, valB);
            }
            return 0;
        } else {
            long valA0 = (long) a[0];
            long valB0 = (long) b[0];
            if (valA0 != valB0) return Long.compare(valA0, valB0);
            for (int i = 1; i < 4; i++) {
                int valA = (int) a[i];
                int valB = (int) b[i];
                if (valA != valB) return Integer.compare(valA, valB);
            }
            return 0;
        }
    }
}
