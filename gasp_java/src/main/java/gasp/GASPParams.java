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
    boolean allowRotation,
    
    // Adaptive Policies
    String updatePolicy,       // "classic", "band", "waste", "adaptive"
    double bandFraction,
    double policyReward,
    double policyDecay,
    
    // 3D Constructives
    boolean layerGreedy,
    boolean useEms,
    boolean parrenoSeed,
    String blockMode
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
            false,
            
            "classic",
            0.05,
            1.0,
            0.99,
            
            false,
            false,
            false,
            "off"
        );
    }
}
