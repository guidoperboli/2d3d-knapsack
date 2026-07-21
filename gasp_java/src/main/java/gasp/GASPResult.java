package gasp;

import gasp.geometry.Packing;
import java.util.List;

/**
 * Result of the GASP metaheuristic.
 */
public record GASPResult(
    Packing bestPacking,
    double bestProfit,
    int iterations,
    double elapsedSeconds,
    List<Double> history
) {}
