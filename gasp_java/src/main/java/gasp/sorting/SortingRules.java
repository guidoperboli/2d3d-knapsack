package gasp.sorting;

import gasp.geometry.Item;
import gasp.geometry.Knapsack;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Sorting criteria used by the Packing Constructive Heuristic (PCH).
 */
public class SortingRules {

    public static final int DELTA_DEFAULT = 10;

    /**
     * Cluster index of value in [lo, hi] using bands of delta%.
     */
    private static int cluster(double value, double lo, double hi, int delta) {
        double span = hi - lo;
        if (span <= 0) return 0;
        double width = span * delta / 100.0;
        if (width <= 0) return 0;
        return (int) Math.floor((value - lo) / width);
    }

    public static List<Item> profitHeight(List<Item> items, Knapsack k, int delta) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c = Double.compare(i2.profit(), i1.profit());
            if (c != 0) return c;
            return Integer.compare(i2.h(), i1.h());
        });
        return sorted;
    }

    public static List<Item> clusteredProfitHeight(List<Item> items, Knapsack k, int delta) {
        if (items.isEmpty()) return new ArrayList<>();
        double pmin = items.stream().mapToDouble(Item::profit).min().orElse(0);
        double pmax = items.stream().mapToDouble(Item::profit).max().orElse(0);
        
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c1 = cluster(i1.profit(), pmin, pmax, delta);
            int c2 = cluster(i2.profit(), pmin, pmax, delta);
            if (c1 != c2) return Integer.compare(c2, c1); // reversed
            return Integer.compare(i2.h(), i1.h()); // reversed
        });
        return sorted;
    }

    public static List<Item> heightProfit(List<Item> items, Knapsack k, int delta) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c = Integer.compare(i2.h(), i1.h());
            if (c != 0) return c;
            return Double.compare(i2.profit(), i1.profit());
        });
        return sorted;
    }

    public static List<Item> clusteredHeightProfit(List<Item> items, Knapsack k, int delta) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c1 = cluster(i1.h(), 0, k.H(), delta);
            int c2 = cluster(i2.h(), 0, k.H(), delta);
            if (c1 != c2) return Integer.compare(c2, c1);
            return Double.compare(i2.profit(), i1.profit());
        });
        return sorted;
    }

    public static List<Item> areaHeight(List<Item> items, Knapsack k, int delta) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c = Integer.compare(i2.baseArea(), i1.baseArea());
            if (c != 0) return c;
            return Integer.compare(i2.h(), i1.h());
        });
        return sorted;
    }

    public static List<Item> clusteredAreaHeight(List<Item> items, Knapsack k, int delta) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c1 = cluster(i1.baseArea(), 0, (double) k.W() * k.D(), delta);
            int c2 = cluster(i2.baseArea(), 0, (double) k.W() * k.D(), delta);
            if (c1 != c2) return Integer.compare(c2, c1);
            return Integer.compare(i2.h(), i1.h());
        });
        return sorted;
    }

    public static List<Item> heightArea(List<Item> items, Knapsack k, int delta) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c = Integer.compare(i2.h(), i1.h());
            if (c != 0) return c;
            return Integer.compare(i2.baseArea(), i1.baseArea());
        });
        return sorted;
    }

    public static List<Item> clusteredHeightArea(List<Item> items, Knapsack k, int delta) {
        List<Item> sorted = new ArrayList<>(items);
        sorted.sort((i1, i2) -> {
            int c1 = cluster(i1.h(), 0, k.H(), delta);
            int c2 = cluster(i2.h(), 0, k.H(), delta);
            if (c1 != c2) return Integer.compare(c2, c1);
            return Integer.compare(i2.baseArea(), i1.baseArea());
        });
        return sorted;
    }

    /** Interface to define a sorting rule. */
    @FunctionalInterface
    public interface SortingRule {
        List<Item> sort(List<Item> items, Knapsack k, int delta);
    }

    public static final List<SortingRule> SORTING_RULES = List.of(
            SortingRules::profitHeight,
            SortingRules::clusteredProfitHeight,
            SortingRules::heightProfit,
            SortingRules::clusteredHeightProfit,
            SortingRules::areaHeight,
            SortingRules::clusteredAreaHeight,
            SortingRules::heightArea,
            SortingRules::clusteredHeightArea
    );
    
    public static boolean isClustered(SortingRule rule) {
        // Simple heuristic: just check identity based on the list
        int idx = SORTING_RULES.indexOf(rule);
        return idx == 1 || idx == 3 || idx == 5 || idx == 7;
    }
}
