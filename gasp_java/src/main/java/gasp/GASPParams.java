package gasp;

/**
 * Configuration parameters for the GASP metaheuristic.
 */
public record GASPParams(
    double timeLimit,          // seconds
    double alpha,              // score decrease
    double beta,               // score increase
    int kInit,                 // initial score multiplier
    int nonImprovingLimit,     // reinit threshold
    int[] pchDeltas,           // delta values for clustered PCH rules
    int reinitSwaps,           // random score swaps in the reinit
    boolean allowRotation
) {
    public static GASPParams defaultParams() {
        return new GASPParams(
            10.0,
            0.2,
            0.1,
            2,
            10,
            new int[]{10},
            3,
            false
        );
    }
}
