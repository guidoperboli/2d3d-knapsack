package gasp.greedy;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Arrays;

import gasp.ems.EMSManager;
import gasp.ems.Space;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;

public class ParrenoConstruct {

    public static class CornerResult {
        public final int[] dist;
        public final int[] sig;

        public CornerResult(int[] dist, int[] sig) {
            this.dist = dist;
            this.sig = sig;
        }
    }

    public static class BlockResult {
        public final List<Placement> placements;
        public final int[] box; // x, y, z, x2, y2, z2

        public BlockResult(List<Placement> placements, int[] box) {
            this.placements = placements;
            this.box = box;
        }
    }
    
    private static class TypeKey {
        int w, d, h;
        public TypeKey(int w, int d, int h) {
            this.w = w; this.d = d; this.h = h;
        }
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            TypeKey typeKey = (TypeKey) o;
            return w == typeKey.w && d == typeKey.d && h == typeKey.h;
        }
        @Override
        public int hashCode() {
            int result = w;
            result = 31 * result + d;
            result = 31 * result + h;
            return result;
        }
    }

    public static CornerResult nearCorner(Space s, Knapsack ks) {
        int W = ks.W();
        int D = ks.D();
        int H = ks.H();

        int mx0 = Math.min(s.x, W - s.x);
        int mx1 = Math.min(s.x2, W - s.x2);
        int my0 = Math.min(s.y, D - s.y);
        int my1 = Math.min(s.y2, D - s.y2);
        int mz0 = Math.min(s.z, H - s.z);
        int mz1 = Math.min(s.z2, H - s.z2);

        int[][][] verts = {
            {{mx0, my0, mz0}, {0, 0, 0}},
            {{mx1, my0, mz0}, {1, 0, 0}},
            {{mx0, my1, mz0}, {0, 1, 0}},
            {{mx0, my0, mz1}, {0, 0, 1}},
            {{mx1, my1, mz0}, {1, 1, 0}},
            {{mx1, my0, mz1}, {1, 0, 1}},
            {{mx0, my1, mz1}, {0, 1, 1}},
            {{mx1, my1, mz1}, {1, 1, 1}}
        };

        int[] bestDist = null;
        int[] bestSig = null;

        for (int[][] vert : verts) {
            int a = vert[0][0];
            int b = vert[0][1];
            int c = vert[0][2];
            int[] sig = vert[1];

            // Sort a, b, c ascending
            if (a > b) { int tmp = a; a = b; b = tmp; }
            if (b > c) { int tmp = b; b = c; c = tmp; }
            if (a > b) { int tmp = a; a = b; b = tmp; }

            int[] v = {a, b, c};
            if (bestDist == null || compareIntArrays(v, bestDist) < 0) {
                bestDist = v;
                bestSig = sig;
            }
        }
        return new CornerResult(bestDist, bestSig);
    }

    private static int compareIntArrays(int[] a, int[] b) {
        for (int i = 0; i < Math.min(a.length, b.length); i++) {
            if (a[i] != b[i]) {
                return Integer.compare(a[i], b[i]);
            }
        }
        return Integer.compare(a.length, b.length);
    }

    public static BlockResult placeBlock(Space s, int[] sig, int w, int d, int h, int nx, int ny, int nz, List<Item> members, int ncopies) {
        int bw = nx * w;
        int bd = ny * d;
        int bh = nz * h;

        int x0 = (sig[0] == 1) ? s.x2 - bw : s.x;
        int y0 = (sig[1] == 1) ? s.y2 - bd : s.y;
        int z0 = (sig[2] == 1) ? s.z2 - bh : s.z;

        List<Placement> pls = new ArrayList<>();
        int ci = 0;
        for (int ix = 0; ix < nx; ix++) {
            for (int iy = 0; iy < ny; iy++) {
                for (int iz = 0; iz < nz; iz++) {
                    if (ci >= ncopies) {
                        break;
                    }
                    Item it = members.get(ci);
                    pls.add(new Placement(it, x0 + ix * w, y0 + iy * d, z0 + iz * h, w, d, h));
                    ci++;
                }
            }
        }
        int[] box = {x0, y0, z0, x0 + bw, y0 + bd, z0 + bh};
        return new BlockResult(pls, box);
    }

    public static List<Space> applyBox(List<Space> spaces, int[] box) {
        return EMSManager.differenceProcess(spaces, box, 1);
    }

    public static Packing parrenoConstruct(List<Item> items, Knapsack ks, boolean allowRotation, String objective) {
        Map<TypeKey, List<Item>> avail = new java.util.LinkedHashMap<>();
        for (Item it : items) {
            TypeKey key = new TypeKey(it.w(), it.d(), it.h());
            avail.computeIfAbsent(key, k -> new ArrayList<>()).add(it);
        }

        List<Space> spaces = new ArrayList<>();
        spaces.add(new Space(0, 0, 0, ks.W(), ks.D(), ks.H()));
        List<Placement> placements = new ArrayList<>();

        while (!spaces.isEmpty()) {
            CornerResult bestCorner = null;
            Space bestSpace = null;
            long bestVol = 0;
            
            for (Space s : spaces) {
                CornerResult cr = nearCorner(s, ks);
                long vol = s.getVolume();
                
                if (bestCorner == null) {
                    bestCorner = cr;
                    bestSpace = s;
                    bestVol = vol;
                } else {
                    int cmp = compareIntArrays(cr.dist, bestCorner.dist);
                    if (cmp < 0 || (cmp == 0 && vol > bestVol)) {
                        bestCorner = cr;
                        bestSpace = s;
                        bestVol = vol;
                    }
                }
            }
            
            Space s = bestSpace;
            int fw = s.getW();
            int fd = s.getD();
            int fh = s.getH();

            Object[] chosenScore = null;
            TypeKey chosenKey = null;
            int[] chosenParams = null; // {w, d, h, nx, ny, nz, ncopies}

            for (Map.Entry<TypeKey, List<Item>> entry : avail.entrySet()) {
                TypeKey tkey = entry.getKey();
                List<Item> members = entry.getValue();
                if (members.isEmpty()) continue;
                int navail = members.size();
                Item rep = members.get(0);

                for (int[] rot : rep.rotations(allowRotation, ks.is3D())) {
                    int w = rot[0], d = rot[1], h = rot[2];
                    if (w > fw || d > fd || h > fh) continue;

                    int maxx = fw / w;
                    int maxy = fd / d;
                    int maxz = fh / h;

                    for (int nx = 1; nx <= maxx; nx++) {
                        for (int ny = 1; ny <= maxy; ny++) {
                            if (nx * ny > navail) break;
                            for (int nz = 1; nz <= maxz; nz++) {
                                int ncopies = nx * ny * nz;
                                if (ncopies > navail) break;

                                int bw = nx * w, bd = ny * d, bh = nz * h;
                                Object[] score;
                                if ("bestvol".equals(objective)) {
                                    score = new Object[]{(long) -(bw * bd * bh), ncopies};
                                } else {
                                    int[] gap = {fw - bw, fd - bd, fh - bh};
                                    Arrays.sort(gap);
                                    score = new Object[]{gap, ncopies};
                                }

                                if (chosenScore == null || compareScores(score, chosenScore, objective) < 0) {
                                    chosenScore = score;
                                    chosenKey = tkey;
                                    chosenParams = new int[]{w, d, h, nx, ny, nz, ncopies};
                                }
                            }
                        }
                    }
                }
            }

            if (chosenParams == null) {
                spaces.remove(s);
                continue;
            }

            int w = chosenParams[0], d = chosenParams[1], h = chosenParams[2];
            int nx = chosenParams[3], ny = chosenParams[4], nz = chosenParams[5], ncopies = chosenParams[6];
            List<Item> members = avail.get(chosenKey);

            BlockResult br = placeBlock(s, bestCorner.sig, w, d, h, nx, ny, nz, members, ncopies);
            placements.addAll(br.placements);
            avail.put(chosenKey, members.subList(ncopies, members.size()));
            spaces = applyBox(spaces, br.box);
        }

        return new Packing(ks, placements);
    }

    private static int compareScores(Object[] a, Object[] b, String objective) {
        if ("bestvol".equals(objective)) {
            long aVol = (long) a[0];
            long bVol = (long) b[0];
            if (aVol != bVol) return Long.compare(aVol, bVol);
        } else {
            int[] aGap = (int[]) a[0];
            int[] bGap = (int[]) b[0];
            int cmp = compareIntArrays(aGap, bGap);
            if (cmp != 0) return cmp;
        }
        int aCopies = (int) a[1];
        int bCopies = (int) b[1];
        return Integer.compare(aCopies, bCopies);
    }
}
