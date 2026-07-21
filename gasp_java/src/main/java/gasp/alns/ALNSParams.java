package gasp.alns;

public record ALNSParams(
    int maxIter,
    double timeLimit,
    double fracLo,
    double fracHi,
    double t0Ratio,
    double cooling,
    int reheatAfter,
    double reheatRatio,
    double react,
    int segUpdate,
    double rewardBest,
    double rewardBetter,
    double rewardAccept,
    boolean allowRotation,
    String objectiveMetric
) {
    public static ALNSParams defaultParams() {
        return new ALNSParams(
            1000, 10.0, 0.1, 0.4, 0.05, 0.995, 200, 2.0, 0.5, 100,
            1.5, 1.2, 0.8, false, "profit"
        );
    }
}
