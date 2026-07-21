package gasp;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;
import gasp.geometry.Packing;
import gasp.greedy.GreedyState;
import gasp.greedy.MeritCriterion;
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
    }

    private Packing place(List<Item> ordered, MeritCriterion criterion) {
        GreedyState state = new GreedyState(knapsack, criterion, p.allowRotation());
        return state.run(ordered);
    }

    public Packing initialSolution() {
        Packing best = null;

        for (SortingRule rule : SortingRules.SORTING_RULES) {
            boolean isClustered = SortingRules.isClustered(rule);
            int[] deltas = isClustered ? p.pchDeltas() : new int[]{10};

            for (int delta : deltas) {
                List<Item> ordered = rule.sort(items, knapsack, delta);
                Packing packing = place(ordered, MeritCriterion.RS);
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

    public void updateScores(Packing current) {
        Set<Integer> loaded = current.loadedIds();
        
        // Update memories
        for (Item item : items) {
            if (loaded.contains(item.idx())) {
                fLoaded.put(item.idx(), fLoaded.get(item.idx()) + 1);
            } else {
                fUnloaded.put(item.idx(), fUnloaded.get(item.idx()) + 1);
            }
        }

        List<Item> loadedItems = new ArrayList<>();
        List<Item> unloadedItems = new ArrayList<>();
        for (Item item : items) {
            if (loaded.contains(item.idx())) loadedItems.add(item);
            else unloadedItems.add(item);
        }

        if (loadedItems.isEmpty() || unloadedItems.isEmpty()) return;

        // least valuable loaded item
        Item jItem = Collections.min(loadedItems, Comparator.comparingDouble(i -> 
            (i.profit() / (double) i.baseArea()) * (1 + fLoaded.get(i.idx()))
        ));

        // most valuable unloaded item
        Item lItem = Collections.max(unloadedItems, Comparator.comparingDouble(i -> 
            i.profit() / ((double) i.baseArea() * (1 + fUnloaded.get(i.idx())))
        ));

        // Swap scores
        double sj = scores.get(jItem.idx()) * (1 - p.alpha());
        double sl = scores.get(lItem.idx()) * (1 + p.beta());
        
        scores.put(jItem.idx(), sl);
        scores.put(lItem.idx(), sj);
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
            
            // Sort by non-increasing score
            List<Item> ordered = new ArrayList<>(items);
            ordered.sort((a, b) -> Double.compare(scores.get(b.idx()), scores.get(a.idx())));
            
            MeritCriterion criterion = MERIT_SEQUENCE[meritIdx];
            Packing current = place(ordered, criterion);

            if (current.profit() > best.profit()) {
                best = current;
                nonImproving = 0;
                k++;
            } else {
                nonImproving++;
            }

            if (nonImproving >= p.nonImprovingLimit()) {
                longTermReinit(best);
                nonImproving = 0;
            } else {
                updateScores(current);
            }

            history.add(best.profit());
        }

        return new GASPResult(best, best.profit(), iterations, (System.currentTimeMillis() - startMs) / 1000.0, history);
    }
}
