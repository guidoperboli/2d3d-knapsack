package gasp.greedy;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.Collections;

public class LayerGreedy {

    private final Knapsack ks;
    private final MeritCriterion criterion;
    private final boolean allowRotation;
    private final int typeThreshold;
    private final double minLayerFill;
    private final int maxSecondTypes;

    public LayerGreedy(Knapsack ks, MeritCriterion criterion, boolean allowRotation, int typeThreshold, double minLayerFill, int maxSecondTypes) {
        this.ks = ks;
        this.criterion = criterion;
        this.allowRotation = allowRotation;
        this.typeThreshold = typeThreshold;
        this.minLayerFill = minLayerFill;
        this.maxSecondTypes = maxSecondTypes;
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

    public Packing run(List<Item> items) {
        Map<TypeKey, List<Item>> types = new java.util.LinkedHashMap<>();
        for (Item it : items) {
            TypeKey key = new TypeKey(it.w(), it.d(), it.h());
            types.computeIfAbsent(key, k -> new ArrayList<>()).add(it);
        }

        if (!ks.is3D() || types.size() <= typeThreshold) {
            return new GreedyState(ks, criterion, allowRotation).run(sorted(items));
        }

        List<Placement> placements = new ArrayList<>();
        Map<TypeKey, List<Item>> avail = new java.util.LinkedHashMap<>(types);
        int zBase = 0;
        int H = ks.H();

        while (zBase < H) {
            int remH = H - zBase;
            LayerChoice choice = pickLayerType(avail, remH);
            if (choice == null) {
                break;
            }
            TypeKey key = choice.key;
            int h = choice.h;
            
            Knapsack slab = new Knapsack(ks.W(), ks.D(), h);
            List<Item> layerItems = new ArrayList<>(avail.get(key));
            List<TypeKey> second = secondType(avail, key, h);
            for (TypeKey k2 : second) {
                layerItems.addAll(avail.get(k2));
            }

            Packing packing = new GreedyState(slab, criterion, allowRotation).run(sorted(layerItems));
            double fill = slab.volume() > 0 ? (double) packing.usedVolume() / slab.volume() : 0.0;
            
            if (packing.getPlacements().isEmpty() || fill < minLayerFill) {
                break;
            }

            Set<Integer> usedIds = new HashSet<>();
            for (Placement p : packing.getPlacements()) {
                usedIds.add(p.item().idx());
                placements.add(new Placement(p.item(), p.x(), p.y(), p.z() + zBase, p.w(), p.d(), p.h()));
            }

            List<TypeKey> toUpdate = new ArrayList<>();
            toUpdate.add(key);
            toUpdate.addAll(second);

            for (TypeKey k : toUpdate) {
                List<Item> remain = new ArrayList<>();
                for (Item it : avail.get(k)) {
                    if (!usedIds.contains(it.idx())) {
                        remain.add(it);
                    }
                }
                if (remain.isEmpty()) {
                    avail.remove(k);
                } else {
                    avail.put(k, remain);
                }
            }

            zBase += h;
            if (avail.isEmpty()) {
                break;
            }
        }

        List<Item> remaining = new ArrayList<>();
        for (List<Item> v : avail.values()) {
            remaining.addAll(v);
        }

        if (!remaining.isEmpty() && zBase < H) {
            Knapsack top = new Knapsack(ks.W(), ks.D(), H - zBase);
            Packing packing = new GreedyState(top, criterion, allowRotation).run(sorted(remaining));
            for (Placement p : packing.getPlacements()) {
                placements.add(new Placement(p.item(), p.x(), p.y(), p.z() + zBase, p.w(), p.d(), p.h()));
            }
        }

        return new Packing(ks, placements);
    }

    private List<Item> sorted(List<Item> items) {
        List<Item> sortedItems = new ArrayList<>(items);
        sortedItems.sort((i1, i2) -> {
            double meas1 = ks.is3D() ? i1.volume() : i1.baseArea();
            double meas2 = ks.is3D() ? i2.volume() : i2.baseArea();
            double val1 = -i1.profit() / meas1;
            double val2 = -i2.profit() / meas2;
            return Double.compare(val1, val2);
        });
        return sortedItems;
    }

    private static class LayerChoice {
        TypeKey key;
        int h;
        LayerChoice(TypeKey k, int h) { this.key = k; this.h = h; }
    }

    private LayerChoice pickLayerType(Map<TypeKey, List<Item>> avail, int remH) {
        long base = (long) ks.W() * ks.D();
        LayerChoice best = null;
        double bestScore = 0.0;

        for (Map.Entry<TypeKey, List<Item>> entry : avail.entrySet()) {
            TypeKey key = entry.getKey();
            List<Item> members = entry.getValue();
            if (members.isEmpty()) continue;
            Item rep = members.get(0);

            for (int[] rot : rep.rotations(allowRotation, ks.is3D())) {
                int w = rot[0], d = rot[1], h = rot[2];
                if (h > remH || w > ks.W() || d > ks.D()) continue;

                int fit = (ks.W() / w) * (ks.D() / d);
                if (fit == 0) continue;

                int copies = Math.min(members.size(), fit);
                double coverage = (double) (copies * w * d) / base;
                if (coverage < 0.35) continue;

                double dens = (double) rep.profit() / rep.volume();
                double score = coverage * dens / h;
                if (score > bestScore) {
                    bestScore = score;
                    best = new LayerChoice(key, h);
                }
            }
        }
        return best;
    }

    private List<Integer> getHeights(Item it) {
        Set<Integer> hs = new HashSet<>();
        for (int[] rot : it.rotations(allowRotation, ks.is3D())) {
            if (rot[0] <= ks.W() && rot[1] <= ks.D() && rot[2] <= ks.H()) {
                hs.add(rot[2]);
            }
        }
        List<Integer> list = new ArrayList<>(hs);
        Collections.sort(list);
        return list;
    }

    private static class SecondTypeScore implements Comparable<SecondTypeScore> {
        double score;
        TypeKey key;
        SecondTypeScore(double s, TypeKey k) { score = s; key = k; }
        @Override
        public int compareTo(SecondTypeScore o) {
            return Double.compare(o.score, this.score); // Reverse order
        }
    }

    private List<TypeKey> secondType(Map<TypeKey, List<Item>> avail, TypeKey key, int h) {
        if (maxSecondTypes <= 0) return new ArrayList<>();
        List<SecondTypeScore> out = new ArrayList<>();
        
        for (Map.Entry<TypeKey, List<Item>> entry : avail.entrySet()) {
            TypeKey k2 = entry.getKey();
            if (k2.equals(key)) continue;
            List<Item> members = entry.getValue();
            if (members.isEmpty()) continue;
            
            Item rep = members.get(0);
            if (getHeights(rep).contains(h)) {
                double score = (double) rep.profit() / rep.volume();
                out.add(new SecondTypeScore(score, k2));
            }
        }
        Collections.sort(out);
        List<TypeKey> result = new ArrayList<>();
        for (int i = 0; i < Math.min(maxSecondTypes, out.size()); i++) {
            result.add(out.get(i).key);
        }
        return result;
    }
}
