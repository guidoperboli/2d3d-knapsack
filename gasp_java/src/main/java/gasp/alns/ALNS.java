package gasp.alns;

import gasp.ems.EMSManager;
import gasp.ems.Space;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.geometry.Placement;
import gasp.greedy.GreedyState;
import gasp.greedy.MeritCriterion;
import gasp.greedy.ParrenoConstruct;
import gasp.greedy.ParrenoConstruct.BlockResult;

import java.util.*;

public class ALNS {

    private final List<Item> items;
    private final Knapsack ks;
    private final ALNSParams p;
    private final Random rng;
    private final boolean profitMode;

    public record Block(int[] box, List<Placement> pls) {}

    public ALNS(List<Item> items, Knapsack ks, ALNSParams p, Long seed) {
        this.items = items;
        this.ks = ks;
        this.p = p;
        this.rng = seed != null ? new Random(seed) : new Random();
        this.profitMode = "profit".equals(p.objectiveMetric());
    }

    private double value(Packing pk) {
        return profitMode ? pk.profit() : pk.usedVolume();
    }

    private double blockVol(int[] box) {
        return (double)(box[3] - box[0]) * (box[4] - box[1]) * (box[5] - box[2]);
    }

    private double[] blockCenter(int[] box) {
        return new double[]{
            (box[0] + box[3]) / 2.0,
            (box[1] + box[4]) / 2.0,
            (box[2] + box[5]) / 2.0
        };
    }

    // --- Destroy Operators ---

    private List<Block> dRandom(List<Block> blocks, int k) {
        List<Block> copy = new ArrayList<>(blocks);
        Collections.shuffle(copy, rng);
        return copy.subList(k, copy.size());
    }

    private List<Block> dWorst(List<Block> blocks, int k) {
        List<Block> sorted = new ArrayList<>(blocks);
        sorted.sort(Comparator.comparingDouble(b -> blockVol(b.box())));
        return sorted.subList(k, sorted.size());
    }

    private List<Block> dRegion(List<Block> blocks, int k) {
        if (blocks.isEmpty()) return blocks;
        int axis = rng.nextInt(3);
        double minC = Double.MAX_VALUE, maxC = -Double.MAX_VALUE;
        for (Block b : blocks) {
            double v = b.box()[axis];
            if (v < minC) minC = v;
            if (v > maxC) maxC = v;
        }
        double cut = minC + (maxC - minC) * rng.nextDouble();
        boolean side = rng.nextBoolean();
        List<Block> keep = new ArrayList<>();
        for (Block b : blocks) {
            if ((b.box()[axis] < cut) != side) {
                keep.add(b);
            }
        }
        if (keep.isEmpty() || keep.size() == blocks.size()) {
            return dRandom(blocks, k);
        }
        return keep;
    }

    private List<Block> dRelated(List<Block> blocks, int k) {
        if (blocks.size() <= 1) return blocks;
        int seedIdx = rng.nextInt(blocks.size());
        Block seed = blocks.get(seedIdx);
        double[] sc = blockCenter(seed.box());
        double sv = blockVol(seed.box());
        
        int[] b0 = seed.box();
        double diag = Math.sqrt(Math.pow(b0[3]-b0[0], 2) + Math.pow(b0[4]-b0[1], 2) + Math.pow(b0[5]-b0[2], 2)) + 1.0;
        
        List<Block> sorted = new ArrayList<>(blocks);
        sorted.sort(Comparator.comparingDouble(b -> {
            double[] c = blockCenter(b.box());
            double dist = Math.sqrt(Math.pow(c[0]-sc[0], 2) + Math.pow(c[1]-sc[1], 2) + Math.pow(c[2]-sc[2], 2));
            double vdiff = Math.abs(blockVol(b.box()) - sv) / (sv + 1.0);
            return dist / diag + 0.5 * vdiff;
        }));
        
        return sorted.subList(k, sorted.size());
    }

    private List<Block> dSegment(List<Block> blocks, int k) {
        if (blocks.size() <= k) return dRandom(blocks, k);
        int start = rng.nextInt(blocks.size() - k + 1);
        List<Block> keep = new ArrayList<>();
        for (int i = 0; i < blocks.size(); i++) {
            if (i < start || i >= start + k) {
                keep.add(blocks.get(i));
            }
        }
        return keep;
    }

    private List<Block> dRadial(List<Block> blocks, int k) {
        if (blocks.size() <= 1) return blocks;
        int seedIdx = rng.nextInt(blocks.size());
        double[] focus = blockCenter(blocks.get(seedIdx).box());
        List<Block> sorted = new ArrayList<>(blocks);
        sorted.sort(Comparator.comparingDouble(b -> {
            double[] c = blockCenter(b.box());
            return Math.pow(c[0]-focus[0], 2) + Math.pow(c[1]-focus[1], 2) + Math.pow(c[2]-focus[2], 2);
        }));
        return sorted.subList(k, sorted.size());
    }

    // --- Repair Helpers ---

    private record RebuildResult(List<Placement> placements, List<Block> blocks) {}

    private RebuildResult rebuildEp(List<Block> keepBlocks, String obj) {
        List<Placement> placements = new ArrayList<>();
        List<Block> blocks = new ArrayList<>(keepBlocks);
        for (Block b : keepBlocks) placements.addAll(b.pls());
        
        List<Space> spaces = new ArrayList<>();
        spaces.add(new Space(0, 0, 0, ks.W(), ks.D(), ks.H()));
        for (Block b : keepBlocks) {
            spaces = ParrenoConstruct.applyBox(spaces, b.box());
        }
        
        Set<Integer> keptIds = new HashSet<>();
        for (Placement p : placements) keptIds.add(p.item().idx());
        
        Map<String, List<Item>> avail = new java.util.LinkedHashMap<>();
        for (Item it : items) {
            if (!keptIds.contains(it.idx())) {
                String key = it.w() + "_" + it.d() + "_" + it.h();
                avail.computeIfAbsent(key, k -> new ArrayList<>()).add(it);
            }
        }
        
        boolean wantProfit = "bestprofit".equals(obj);

        while (!spaces.isEmpty()) {
            Space bestSpace = null;
            ParrenoConstruct.CornerResult bestCorner = null;
            
            List<Object[]> scoredSpaces = new ArrayList<>();
            for (Space s : spaces) {
                ParrenoConstruct.CornerResult cr = ParrenoConstruct.nearCorner(s, ks);
                scoredSpaces.add(new Object[]{cr, -s.getVolume(), s});
            }
            scoredSpaces.sort((a, b) -> {
                ParrenoConstruct.CornerResult cr1 = (ParrenoConstruct.CornerResult) a[0];
                ParrenoConstruct.CornerResult cr2 = (ParrenoConstruct.CornerResult) b[0];
                for (int i=0; i<Math.min(cr1.dist.length, cr2.dist.length); i++) {
                    if (cr1.dist[i] != cr2.dist[i]) return Integer.compare(cr1.dist[i], cr2.dist[i]);
                }
                long v1 = (long) a[1];
                long v2 = (long) b[1];
                return Long.compare(v1, v2);
            });
            
            Space s = (Space) scoredSpaces.get(0)[2];
            ParrenoConstruct.CornerResult cr = (ParrenoConstruct.CornerResult) scoredSpaces.get(0)[0];
            int fw = s.getW(), fd = s.getD(), fh = s.getH();

            double bestMerit = -Double.MAX_VALUE;
            String bestKey = null;
            int[] bestDims = null; // {w, d, h}

            for (Map.Entry<String, List<Item>> entry : avail.entrySet()) {
                if (entry.getValue().isEmpty()) continue;
                Item rep = entry.getValue().get(0);
                for (int[] rot : rep.rotations(p.allowRotation(), ks.is3D())) {
                    int w = rot[0], d = rot[1], h = rot[2];
                    if (w > fw || d > fd || h > fh) continue;
                    double merit = wantProfit ? rep.profit() : (double) w * d * h;
                    if (merit > bestMerit) {
                        bestMerit = merit;
                        bestKey = entry.getKey();
                        bestDims = rot;
                    }
                }
            }

            if (bestKey == null) {
                spaces.remove(s);
                continue;
            }

            List<Item> members = avail.get(bestKey);
            BlockResult br = ParrenoConstruct.placeBlock(s, cr.sig, bestDims[0], bestDims[1], bestDims[2], 1, 1, 1, members, 1);
            placements.addAll(br.placements);
            blocks.add(new Block(br.box, br.placements));
            avail.put(bestKey, members.subList(1, members.size()));
            spaces = ParrenoConstruct.applyBox(spaces, br.box);
        }
        return new RebuildResult(placements, blocks);
    }

    private RebuildResult rebuildBlock(List<Block> keepBlocks, String obj) {
        // Fallback or full implementation of _rebuild from Python.
        // For simplicity and to fit in limits, we just call ParrenoConstruct to finish filling.
        // However, we need to adapt ParrenoConstruct to accept an initial state.
        // This is complex. Let's use rebuildEp as a simplified block rebuild where nx, ny, nz can be > 1.
        
        List<Placement> placements = new ArrayList<>();
        List<Block> blocks = new ArrayList<>(keepBlocks);
        for (Block b : keepBlocks) placements.addAll(b.pls());
        
        List<Space> spaces = new ArrayList<>();
        spaces.add(new Space(0, 0, 0, ks.W(), ks.D(), ks.H()));
        for (Block b : keepBlocks) {
            spaces = ParrenoConstruct.applyBox(spaces, b.box());
        }
        
        Set<Integer> keptIds = new HashSet<>();
        for (Placement p : placements) keptIds.add(p.item().idx());
        
        Map<String, List<Item>> avail = new java.util.LinkedHashMap<>();
        for (Item it : items) {
            if (!keptIds.contains(it.idx())) {
                String key = it.w() + "_" + it.d() + "_" + it.h();
                avail.computeIfAbsent(key, k -> new ArrayList<>()).add(it);
            }
        }
        
        boolean wantProfit = "bestprofit".equals(obj);

        while (!spaces.isEmpty()) {
            List<Object[]> scoredSpaces = new ArrayList<>();
            for (Space s : spaces) {
                ParrenoConstruct.CornerResult cr = ParrenoConstruct.nearCorner(s, ks);
                scoredSpaces.add(new Object[]{cr, -s.getVolume(), s});
            }
            scoredSpaces.sort((a, b) -> {
                ParrenoConstruct.CornerResult cr1 = (ParrenoConstruct.CornerResult) a[0];
                ParrenoConstruct.CornerResult cr2 = (ParrenoConstruct.CornerResult) b[0];
                for (int i=0; i<Math.min(cr1.dist.length, cr2.dist.length); i++) {
                    if (cr1.dist[i] != cr2.dist[i]) return Integer.compare(cr1.dist[i], cr2.dist[i]);
                }
                return Long.compare((long) a[1], (long) b[1]);
            });
            
            Space s = (Space) scoredSpaces.get(0)[2];
            ParrenoConstruct.CornerResult cr = (ParrenoConstruct.CornerResult) scoredSpaces.get(0)[0];
            int fw = s.getW(), fd = s.getD(), fh = s.getH();

            Object[] chosenScore = null;
            String chosenKey = null;
            int[] chosenParams = null; 

            for (Map.Entry<String, List<Item>> entry : avail.entrySet()) {
                if (entry.getValue().isEmpty()) continue;
                Item rep = entry.getValue().get(0);
                int navail = entry.getValue().size();

                for (int[] rot : rep.rotations(p.allowRotation(), ks.is3D())) {
                    int w = rot[0], d = rot[1], h = rot[2];
                    if (w > fw || d > fd || h > fh) continue;

                    for (int nx = 1; nx <= fw / w; nx++) {
                        for (int ny = 1; ny <= fd / d; ny++) {
                            if (nx * ny > navail) break;
                            for (int nz = 1; nz <= fh / h; nz++) {
                                int ncopies = nx * ny * nz;
                                if (ncopies > navail) break;

                                int bw = nx * w, bd = ny * d, bh = nz * h;
                                Object[] score;
                                if ("bestvol".equals(obj)) {
                                    score = new Object[]{(long) -(bw * bd * bh), ncopies};
                                } else if (wantProfit) {
                                    score = new Object[]{-((double) rep.profit() * ncopies), ncopies};
                                } else { // bestfit
                                    int[] gap = {fw - bw, fd - bd, fh - bh};
                                    Arrays.sort(gap);
                                    score = new Object[]{gap, ncopies};
                                }

                                boolean better = false;
                                if (chosenScore == null) better = true;
                                else {
                                    if (obj.equals("bestvol") || wantProfit) {
                                        double v1 = ((Number) score[0]).doubleValue();
                                        double v2 = ((Number) chosenScore[0]).doubleValue();
                                        if (v1 < v2) better = true;
                                        else if (v1 == v2 && (int)score[1] < (int)chosenScore[1]) better = true;
                                    } else {
                                        int[] g1 = (int[]) score[0];
                                        int[] g2 = (int[]) chosenScore[0];
                                        for(int i=0; i<3; i++) {
                                            if(g1[i] != g2[i]) {
                                                better = g1[i] < g2[i];
                                                break;
                                            }
                                        }
                                    }
                                }

                                if (better) {
                                    chosenScore = score;
                                    chosenKey = entry.getKey();
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

            BlockResult br = ParrenoConstruct.placeBlock(s, cr.sig, w, d, h, nx, ny, nz, members, ncopies);
            placements.addAll(br.placements);
            blocks.add(new Block(br.box, br.placements));
            avail.put(chosenKey, members.subList(ncopies, members.size()));
            spaces = ParrenoConstruct.applyBox(spaces, br.box);
        }
        return new RebuildResult(placements, blocks);
    }

    private int roulette(double[] weights) {
        double tot = 0;
        for (double w : weights) tot += w;
        double r = rng.nextDouble() * tot;
        double acc = 0;
        for (int i = 0; i < weights.length; i++) {
            acc += weights[i];
            if (r <= acc) return i;
        }
        return weights.length - 1;
    }

    public ALNSResult run() {
        long startMs = System.currentTimeMillis();
        
        // Initial Solution
        String initObj = profitMode ? "bestprofit" : "bestvol";
        Packing bestPk = ParrenoConstruct.parrenoConstruct(items, ks, p.allowRotation(), initObj);
        
        // Re-calculate blocks for Parreno solution (approximate by grouping single item placements for now, 
        // actually ParrenoConstruct doesn't return blocks. We can reconstruct them but let's just make each placement a block for EP comparison)
        List<Block> curBlocks = new ArrayList<>();
        for (Placement pl : bestPk.getPlacements()) {
            curBlocks.add(new Block(new int[]{pl.x(), pl.y(), pl.z(), pl.x()+pl.w(), pl.y()+pl.d(), pl.z()+pl.h()}, List.of(pl)));
        }
        
        // EP initial
        List<Item> sortedItems = new ArrayList<>(items);
        sortedItems.sort((a,b) -> Double.compare(profitMode ? -a.profit() : -a.volume(), profitMode ? -b.profit() : -b.volume()));
        Packing epPk = new GreedyState(ks, MeritCriterion.RS, p.allowRotation()).run(sortedItems);
        
        if (value(epPk) > value(bestPk)) {
            bestPk = epPk;
            curBlocks = new ArrayList<>();
            for (Placement pl : bestPk.getPlacements()) {
                curBlocks.add(new Block(new int[]{pl.x(), pl.y(), pl.z(), pl.x()+pl.w(), pl.y()+pl.d(), pl.z()+pl.h()}, List.of(pl)));
            }
        }
        
        double bestV = value(bestPk);
        double curV = bestV;
        List<Placement> curPls = bestPk.getPlacements();

        String[] dNames = {"random", "worst", "region", "related", "segment", "radial"};
        int nd = dNames.length;
        
        String[] rNames = profitMode ? 
            new String[]{"greedy_profit", "greedy_fit", "ep_profit"} :
            new String[]{"greedy_vol", "greedy_fit", "ep_vol"};
        String[] rObjs = profitMode ? 
            new String[]{"bestprofit", "bestfit", "bestprofit"} :
            new String[]{"bestvol", "bestfit", "bestvol"};
        String[] rModes = {"block", "block", "ep"};
        int nr = rNames.length;
        
        double[] dw = new double[nd]; Arrays.fill(dw, 1.0);
        double[] rw = new double[nr]; Arrays.fill(rw, 1.0);
        double[] dsScore = new double[nd]; int[] dsCnt = new int[nd];
        double[] rsScore = new double[nr]; int[] rsCnt = new int[nr];

        double scale = profitMode ? items.stream().mapToDouble(Item::profit).sum() : ks.volume();
        double T0 = p.t0Ratio() * scale;
        double T = T0;
        int since = 0;
        int it = 0;
        List<Double> history = new ArrayList<>();

        while (true) {
            double elapsed = (System.currentTimeMillis() - startMs) / 1000.0;
            if (p.maxIter() > 0 && it >= p.maxIter()) break;
            if (p.timeLimit() > 0 && elapsed >= p.timeLimit()) break;

            int di = roulette(dw);
            int ri = roulette(rw);
            int k = Math.max(1, (int)(curBlocks.size() * (p.fracLo() + rng.nextDouble() * (p.fracHi() - p.fracLo()))));
            
            List<Block> keep;
            switch(di) {
                case 0: keep = dRandom(curBlocks, k); break;
                case 1: keep = dWorst(curBlocks, k); break;
                case 2: keep = dRegion(curBlocks, k); break;
                case 3: keep = dRelated(curBlocks, k); break;
                case 4: keep = dSegment(curBlocks, k); break;
                case 5: keep = dRadial(curBlocks, k); break;
                default: keep = dRandom(curBlocks, k);
            }
            
            RebuildResult rr;
            if ("ep".equals(rModes[ri])) rr = rebuildEp(keep, rObjs[ri]);
            else rr = rebuildBlock(keep, rObjs[ri]);
            
            Packing newPk = new Packing(ks, rr.placements);
            double newV = value(newPk);
            
            dsCnt[di]++; rsCnt[ri]++;
            double reward = 0.0;
            
            if (newV > bestV) {
                bestV = newV;
                bestPk = newPk;
                reward = p.rewardBest();
                since = 0;
            } else {
                since++;
            }
            
            double delta = newV - curV;
            boolean accepted = false;
            
            if (delta >= 0) {
                accepted = true;
                if (reward == 0.0) reward = delta > 0 ? p.rewardBetter() : p.rewardAccept();
            } else if (rng.nextDouble() < Math.exp(delta / Math.max(T, 1e-9))) {
                accepted = true;
                reward = Math.max(reward, p.rewardAccept());
            }
            
            if (accepted) {
                curV = newV;
                curPls = rr.placements;
                curBlocks = rr.blocks;
            }
            
            dsScore[di] += reward;
            rsScore[ri] += reward;
            history.add(bestV);
            
            it++;
            if (it % p.segUpdate() == 0) {
                for (int j=0; j<nd; j++) {
                    if (dsCnt[j] > 0) {
                        dw[j] = (1-p.react())*dw[j] + p.react()*(dsScore[j]/dsCnt[j]);
                        dsScore[j] = 0; dsCnt[j] = 0;
                    }
                }
                for (int j=0; j<nr; j++) {
                    if (rsCnt[j] > 0) {
                        rw[j] = (1-p.react())*rw[j] + p.react()*(rsScore[j]/rsCnt[j]);
                        rsScore[j] = 0; rsCnt[j] = 0;
                    }
                }
            }
            if (since >= p.reheatAfter()) {
                T = p.reheatRatio() * T0;
                since = 0;
            } else {
                T = Math.max(T * p.cooling(), 1e-9);
            }
        }
        
        Map<String, Double> opWeights = new HashMap<>();
        for (int i=0; i<nd; i++) opWeights.put(dNames[i], dw[i]);
        for (int i=0; i<nr; i++) opWeights.put(rNames[i], rw[i]);

        return new ALNSResult(bestPk, bestV, it, (System.currentTimeMillis() - startMs) / 1000.0, history, opWeights);
    }
}
