package gasp;

import gasp.ep.EPManager;
import gasp.ep.ExtremePoint;
import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.greedy.GreedyState;
import gasp.greedy.EMSGreedy;
import gasp.greedy.LayerGreedy;
import gasp.greedy.MeritCriterion;
import gasp.greedy.ParrenoConstruct;
import gasp.sorting.SortingRules;
import gasp.sorting.SortingRules.SortingRule;

import java.util.*;

/**
 * GASP - Greedy Adaptive Search Procedure.
 */
public class GASP {

    private static final MeritCriterion[] MERIT_SEQUENCE = {
        MeritCriterion.RS, MeritCriterion.MP, MeritCriterion.LEV, MeritCriterion.FF
    };

    private final List<Item> items;
    private final Knapsack knapsack;
    private final GASPParams p;
    private final Random rng;

    private final Map<Integer, Double> scores = new HashMap<>();
    private int k;
    
    // Long-term memories
    private final Map<Integer, Integer> fLoaded = new HashMap<>();
    private final Map<Integer, Integer> fUnloaded = new HashMap<>();
    
    private int meritIdx = 0;
    
    // Adaptive policy weights
    private final Map<String, Double> policyWeights = new HashMap<>();

    public GASP(List<Item> items, Knapsack knapsack, GASPParams params, Long seed) {
        this.items = items;
        this.knapsack = knapsack;
        this.p = params;
        this.rng = seed != null ? new Random(seed) : new Random();

        for (Item item : items) {
            fLoaded.put(item.idx(), 0);
            fUnloaded.put(item.idx(), 0);
        }
        this.k = params.kInit();
        
        policyWeights.put("classic", 1.0);
        policyWeights.put("band", 1.0);
        policyWeights.put("waste", 1.0);
    }

    private static class PlaceResult {
        public final Packing packing;
        public final EPManager epm;
        public PlaceResult(Packing packing, EPManager epm) {
            this.packing = packing;
            this.epm = epm;
        }
    }

    private PlaceResult place(List<Item> ordered, MeritCriterion criterion, boolean returnEps) {
        if (p.useEms() && knapsack.is3D()) {
            String crit = (criterion == MeritCriterion.RS || criterion == MeritCriterion.TP) ? "BSS" : "VOL";
            Packing pk = new EMSGreedy(knapsack, crit, p.allowRotation()).run(ordered);
            return new PlaceResult(pk, null);
        }
        
        // TODO: block_mode not fully implemented yet in Java, falling back to basic GreedyState
        GreedyState state = new GreedyState(knapsack, criterion, p.allowRotation());
        Packing pk = state.run(ordered);
        return new PlaceResult(pk, returnEps ? state.getEpManager() : null);
    }

    public Packing initialSolution() {
        Packing best = null;

        if (p.parrenoSeed() && knapsack.is3D()) {
            best = ParrenoConstruct.parrenoConstruct(items, knapsack, p.allowRotation(), "bestvol");
        }
        
        if (p.layerGreedy() && knapsack.is3D()) {
            LayerGreedy lg = new LayerGreedy(knapsack, MeritCriterion.RS, p.allowRotation(), 6, 0.45, 2);
            Packing cand = lg.run(items);
            if (best == null || cand.profit() > best.profit()) {
                best = cand;
            }
        }

        for (SortingRule rule : SortingRules.SORTING_RULES) {
            boolean isClustered = SortingRules.isClustered(rule);
            int[] deltas = isClustered ? p.pchDeltas() : new int[]{10};

            for (int delta : deltas) {
                List<Item> ordered = rule.sort(items, knapsack, delta);
                Packing packing = place(ordered, MeritCriterion.RS, false).packing;
                if (best == null || packing.profit() > best.profit()) {
                    best = packing;
                }
            }
        }
        return best;
    }

    public void initScores(Packing reference, int kVal) {
        Set<Integer> loaded = reference.loadedIds();
        for (Item item : items) {
            double s = loaded.contains(item.idx()) ? kVal * item.profit() : item.profit();
            scores.put(item.idx(), s);
        }
    }

    public void updateScores(Packing current, boolean updateMemories) {
        Set<Integer> loaded = current.loadedIds();
        
        if (updateMemories) {
            for (Item item : items) {
                if (loaded.contains(item.idx())) {
                    fLoaded.put(item.idx(), fLoaded.get(item.idx()) + 1);
                } else {
                    fUnloaded.put(item.idx(), fUnloaded.get(item.idx()) + 1);
                }
            }
        }

        List<Item> loadedItems = new ArrayList<>();
        List<Item> unloadedItems = new ArrayList<>();
        for (Item item : items) {
            if (loaded.contains(item.idx())) loadedItems.add(item);
            else unloadedItems.add(item);
        }

        if (loadedItems.isEmpty() || unloadedItems.isEmpty()) return;

        Item jItem = Collections.min(loadedItems, Comparator.comparingDouble(i -> 
            (i.profit() / (double) i.baseArea()) * (1 + fLoaded.get(i.idx()))
        ));

        Item lItem = Collections.max(unloadedItems, Comparator.comparingDouble(i -> 
            i.profit() / ((double) i.baseArea() * (1 + fUnloaded.get(i.idx())))
        ));

        double sj = scores.get(jItem.idx()) * (1 - p.alpha());
        double sl = scores.get(lItem.idx()) * (1 + p.beta());
        
        scores.put(jItem.idx(), sl);
        scores.put(lItem.idx(), sj);
    }
    
    public void updateScoresBand(Packing current) {
        Set<Integer> loaded = current.loadedIds();
        List<Item> loadedItems = new ArrayList<>();
        List<Item> unloadedItems = new ArrayList<>();
        for (Item item : items) {
            if (loaded.contains(item.idx())) loadedItems.add(item);
            else unloadedItems.add(item);
        }
        if (loadedItems.isEmpty() || unloadedItems.isEmpty()) return;
        
        int b = Math.max(2, (int) (items.size() * p.bandFraction()));
        
        loadedItems.sort(Comparator.comparingDouble(i -> 
            (i.profit() / (double) i.baseArea()) * (1 + fLoaded.get(i.idx()))
        ));
        
        unloadedItems.sort((i1, i2) -> Double.compare(
            -i2.profit() / ((double) i2.baseArea() * (1 + fUnloaded.get(i2.idx()))),
            -i1.profit() / ((double) i1.baseArea() * (1 + fUnloaded.get(i1.idx())))
        ));
        
        List<Item> worstIn = new ArrayList<>(loadedItems.subList(0, Math.min(b, loadedItems.size())));
        List<Item> bestOut = new ArrayList<>(unloadedItems.subList(0, Math.min(b, unloadedItems.size())));
        
        int nPairs = rng.nextInt(Math.min(worstIn.size(), bestOut.size())) + 1;
        Collections.shuffle(worstIn, rng);
        Collections.shuffle(bestOut, rng);
        
        for (int i = 0; i < nPairs; i++) {
            Item jItem = worstIn.get(i);
            Item lItem = bestOut.get(i);
            double sj = scores.get(jItem.idx()) * (1 - p.alpha());
            double sl = scores.get(lItem.idx()) * (1 + p.beta());
            scores.put(jItem.idx(), sl);
            scores.put(lItem.idx(), sj);
        }
    }
    
    public void updateScoresWaste(Packing current, EPManager epm) {
        Set<Integer> loaded = current.loadedIds();
        List<Item> loadedItems = new ArrayList<>();
        List<Item> unloadedItems = new ArrayList<>();
        for (Item item : items) {
            if (loaded.contains(item.idx())) loadedItems.add(item);
            else unloadedItems.add(item);
        }
        if (loadedItems.isEmpty() || unloadedItems.isEmpty() || epm == null) return;
        
        List<ExtremePoint> freeBoxes = new ArrayList<>(epm.getEps());
        freeBoxes.sort((ep1, ep2) -> Long.compare(
            (long) ep2.getRsX() * ep2.getRsY() * ep2.getRsZ(),
            (long) ep1.getRsX() * ep1.getRsY() * ep1.getRsZ()
        ));
        freeBoxes = freeBoxes.subList(0, Math.min(5, freeBoxes.size()));
        if (freeBoxes.isEmpty()) return;
        
        List<Item> candidates = new ArrayList<>();
        for (Item item : unloadedItems) {
            boolean fits = false;
            for (ExtremePoint ep : freeBoxes) {
                for (int[] rot : item.rotations(p.allowRotation(), knapsack.is3D())) {
                    if (rot[0] <= ep.getRsX() && rot[1] <= ep.getRsY() && rot[2] <= ep.getRsZ()) {
                        fits = true;
                        break;
                    }
                }
                if (fits) break;
            }
            if (fits) candidates.add(item);
        }
        
        if (candidates.isEmpty()) return;
        
        candidates.sort((i1, i2) -> Double.compare(-i1.profit() / (double) i1.volume(), -i2.profit() / (double) i2.volume()));
        loadedItems.sort(Comparator.comparingDouble(i -> i.profit() / (double) i.volume()));
        
        int nSwap = Math.min(3, Math.min(candidates.size(), loadedItems.size()));
        for (int i = 0; i < nSwap; i++) {
            Item jItem = loadedItems.get(i);
            Item lItem = candidates.get(i);
            double sj = scores.get(jItem.idx()) * (1 - p.alpha());
            double sl = scores.get(lItem.idx()) * (1 + p.beta());
            scores.put(jItem.idx(), sl);
            scores.put(lItem.idx(), sj);
        }
    }
    
    private String selectPolicy() {
        if (!"adaptive".equals(p.updatePolicy())) {
            return p.updatePolicy();
        }
        double total = policyWeights.values().stream().mapToDouble(Double::doubleValue).sum();
        double r = rng.nextDouble() * total;
        double acc = 0.0;
        for (Map.Entry<String, Double> entry : policyWeights.entrySet()) {
            acc += entry.getValue();
            if (r <= acc) {
                return entry.getKey();
            }
        }
        return "classic";
    }

    public void longTermReinit(Packing best) {
        initScores(best, p.kInit());
        List<Integer> ids = new ArrayList<>();
        for (Item item : items) ids.add(item.idx());
        
        for (int i = 0; i < p.reinitSwaps(); i++) {
            int a = ids.get(rng.nextInt(ids.size()));
            int b = ids.get(rng.nextInt(ids.size()));
            double temp = scores.get(a);
            scores.put(a, scores.get(b));
            scores.put(b, temp);
        }
        
        meritIdx = (meritIdx + 1) % MERIT_SEQUENCE.length;
        k = 1;
    }

    public GASPResult run() {
        long startMs = System.currentTimeMillis();

        Packing best = initialSolution();
        initScores(best, p.kInit());
        List<Double> history = new ArrayList<>();
        history.add(best.profit());

        int iterations = 0;
        int nonImproving = 0;

        while (true) {
            double elapsed = (System.currentTimeMillis() - startMs) / 1000.0;
            if (elapsed >= p.timeLimit()) {
                break;
            }

            iterations++;
            
            List<Item> ordered = new ArrayList<>(items);
            ordered.sort((a, b) -> Double.compare(scores.get(b.idx()), scores.get(a.idx())));
            
            MeritCriterion criterion = MERIT_SEQUENCE[meritIdx];
            String policy = selectPolicy();
            boolean needEps = "waste".equals(policy);
            
            PlaceResult pr = place(ordered, criterion, needEps);
            Packing current = pr.packing;
            EPManager epm = pr.epm;

            if (current.profit() > best.profit()) {
                best = current;
                nonImproving = 0;
                k++;
                if ("adaptive".equals(p.updatePolicy())) {
                    policyWeights.put(policy, policyWeights.get(policy) + p.policyReward());
                }
            } else {
                nonImproving++;
                if ("adaptive".equals(p.updatePolicy())) {
                    policyWeights.put(policy, Math.max(0.1, policyWeights.get(policy) * p.policyDecay()));
                }
            }

            if (nonImproving >= p.nonImprovingLimit()) {
                longTermReinit(best);
                nonImproving = 0;
            } else {
                // Shared long-term memories update
                Set<Integer> loaded = current.loadedIds();
                for (Item item : items) {
                    if (loaded.contains(item.idx())) {
                        fLoaded.put(item.idx(), fLoaded.get(item.idx()) + 1);
                    } else {
                        fUnloaded.put(item.idx(), fUnloaded.get(item.idx()) + 1);
                    }
                }
                
                if ("band".equals(policy)) {
                    updateScoresBand(current);
                } else if ("waste".equals(policy)) {
                    updateScoresWaste(current, epm);
                } else {
                    updateScores(current, false); // memories already updated
                }
            }

            history.add(best.profit());
        }

        return new GASPResult(best, best.profit(), iterations, (System.currentTimeMillis() - startMs) / 1000.0, history);
    }
}
