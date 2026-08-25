package gasp.alns;

import gasp.geometry.Packing;
import java.util.List;
import java.util.Map;

public record ALNSResult(
    Packing bestPacking,
    double bestProfit,
    int iterations,
    double elapsedSeconds,
    double timeToBest,
    List<Double> history,
    Map<String, Double> opWeights
) {}
