package gasp.geometry;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A rectangular (boxed) item.
 * <p>
 * Implemented as a record for immutability and memory efficiency.
 *
 * @param idx    unique index of the item in the instance
 * @param w      width (x)
 * @param d      depth (y)
 * @param h      height (z)
 * @param profit profit gained when the item is loaded
 * @param vw     whether the width dimension may be placed vertically (along the container height)
 * @param vd     whether the depth dimension may be placed vertically
 * @param vh     whether the height dimension may be placed vertically
 */
public record Item(int idx, int w, int d, int h, double profit, boolean vw, boolean vd, boolean vh, boolean hasVFlags) {

    // Overloaded constructor for when vflags is None/null in Python
    public Item(int idx, int w, int d, int h, double profit) {
        this(idx, w, d, h, profit, true, true, true, false);
    }

    public int volume() {
        return w * d * h;
    }

    public int baseArea() {
        return w * d;
    }

    // Process-wide cache of rotation results
    private static final Map<RotationKey, List<int[]>> ROT_CACHE = new ConcurrentHashMap<>();

    // The 6 axis permutations used to enumerate 3D orientations
    private static final int[][] AXIS_PERMS = {
            {0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
    };

    private record RotationKey(int w, int d, int h, boolean vw, boolean vd, boolean vh, boolean hasVFlags, boolean allowRotation, boolean is3D) {}

    /**
     * Return the list of distinct (w, d, h) orientations.
     *
     * @param allowRotation whether rotation is allowed
     * @param is3D          whether the problem is 3D
     * @return List of arrays, where each array is [w, d, h]
     */
    public List<int[]> rotations(boolean allowRotation, boolean is3D) {
        RotationKey key = new RotationKey(w, d, h, vw, vd, vh, hasVFlags, allowRotation, is3D);

        return ROT_CACHE.computeIfAbsent(key, k -> {
            int[] dims = {w, d, h};
            if (!allowRotation) {
                return List.of(dims);
            }

            if (is3D) {
                Set<List<Integer>> seen = new HashSet<>();
                List<int[]> rots = new ArrayList<>();
                for (int[] perm : AXIS_PERMS) {
                    int candW = dims[perm[0]];
                    int candD = dims[perm[1]];
                    int candH = dims[perm[2]];

                    if (hasVFlags) {
                        boolean canBeVertical = switch (perm[2]) {
                            case 0 -> vw;
                            case 1 -> vd;
                            case 2 -> vh;
                            default -> throw new IllegalStateException("Unexpected value: " + perm[2]);
                        };
                        if (!canBeVertical) {
                            continue;
                        }
                    }

                    List<Integer> candList = List.of(candW, candD, candH);
                    if (seen.add(candList)) {
                        rots.add(new int[]{candW, candD, candH});
                    }
                }
                
                // Sort to guarantee determinism
                rots.sort(Comparator.<int[]>comparingInt(a -> a[0])
                        .thenComparingInt(a -> a[1])
                        .thenComparingInt(a -> a[2]));
                
                return rots.isEmpty() ? List.of(dims) : List.copyOf(rots);
            } else {
                List<int[]> rots = new ArrayList<>();
                rots.add(dims);
                if (w != d) {
                    rots.add(new int[]{d, w, h});
                }
                rots.sort(Comparator.<int[]>comparingInt(a -> a[0]).thenComparingInt(a -> a[1]));
                return List.copyOf(rots);
            }
        });
    }
}
