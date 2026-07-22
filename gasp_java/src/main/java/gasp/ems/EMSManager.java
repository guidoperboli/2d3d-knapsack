package gasp.ems;

import gasp.geometry.Placement;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class EMSManager {

    /**
     * Aggiorna la lista di EMS dopo il piazzamento di un box.
     */
    public static List<Space> differenceProcess(List<Space> spaces, Placement p, int minDim) {
        return differenceProcess(spaces, p.x(), p.y(), p.z(), p.x2(), p.y2(), p.z2(), minDim);
    }
    
    /**
     * Aggiorna la lista di EMS dopo il piazzamento di un box generico.
     */
    public static List<Space> differenceProcess(List<Space> spaces, int bx, int by, int bz, int bx2, int by2, int bz2, int minDim) {
        List<Space> newSpaces = new ArrayList<>();
        for (Space s : spaces) {
            if (overlapsBox(s, bx, by, bz, bx2, by2, bz2)) {
                splitOne(s, bx, by, bz, bx2, by2, bz2, newSpaces);
            } else {
                newSpaces.add(s);
            }
        }
        return removeDominated(newSpaces, minDim);
    }

    /**
     * Metodo di utility se si passa un array int[6]
     */
    public static List<Space> differenceProcess(List<Space> spaces, int[] box, int minDim) {
        return differenceProcess(spaces, box[0], box[1], box[2], box[3], box[4], box[5], minDim);
    }

    private static boolean overlapsBox(Space s, int bx, int by, int bz, int bx2, int by2, int bz2) {
        return !(s.x2 <= bx || bx2 <= s.x ||
                 s.y2 <= by || by2 <= s.y ||
                 s.z2 <= bz || bz2 <= s.z);
    }

    private static void splitOne(Space s, int bx, int by, int bz, int bx2, int by2, int bz2, List<Space> out) {
        // along X
        if (bx > s.x) {
            out.add(new Space(s.x, s.y, s.z, bx, s.y2, s.z2));
        }
        if (bx2 < s.x2) {
            out.add(new Space(bx2, s.y, s.z, s.x2, s.y2, s.z2));
        }
        // along Y
        if (by > s.y) {
            out.add(new Space(s.x, s.y, s.z, s.x2, by, s.z2));
        }
        if (by2 < s.y2) {
            out.add(new Space(s.x, by2, s.z, s.x2, s.y2, s.z2));
        }
        // along Z
        if (bz > s.z) {
            out.add(new Space(s.x, s.y, s.z, s.x2, s.y2, bz));
        }
        if (bz2 < s.z2) {
            out.add(new Space(s.x, s.y, bz2, s.x2, s.y2, s.z2));
        }
    }

    /**
     * Rimuove spazi contenuti all'interno di altri spazi o troppo piccoli.
     */
    public static List<Space> removeDominated(List<Space> spaces, int minDim) {
        List<Space> cand = new ArrayList<>(spaces.size());
        for (Space s : spaces) {
            if (s.getW() >= minDim && s.getD() >= minDim && s.getH() >= minDim) {
                cand.add(s);
            }
        }
        
        int n = cand.size();
        if (n < 2) return cand;
        
        boolean[] keep = new boolean[n];
        for (int i = 0; i < n; i++) keep[i] = true;
        
        for (int i = 0; i < n; i++) {
            if (!keep[i]) continue;
            Space si = cand.get(i);
            for (int j = 0; j < n; j++) {
                if (i == j || !keep[j]) continue;
                Space sj = cand.get(j);
                if (si.containsFast(sj.x, sj.y, sj.z, sj.x2, sj.y2, sj.z2)) {
                    keep[j] = false;
                }
            }
        }
        
        List<Space> result = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (keep[i]) result.add(cand.get(i));
        }
        return result;
    }
}
