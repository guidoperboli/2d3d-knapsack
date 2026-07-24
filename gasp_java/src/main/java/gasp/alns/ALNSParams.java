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
            -1, 10.0, 0.15, 0.45, 0.02, 0.9995, 1500, 0.5, 0.1, 25,
            4.0, 2.0, 1.0, true, "volume"
        );
    }
}
