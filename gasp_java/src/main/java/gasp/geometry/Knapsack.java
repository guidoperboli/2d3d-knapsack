package gasp.geometry;

/**
 * The single large object (knapsack / container).
 * <p>
 * Implemented as a record.
 */
public record Knapsack(int W, int D, int H) {

    // Overloaded constructor for 2D problems (where H = 1)
    public Knapsack(int W, int D) {
        this(W, D, 1);
    }

    public boolean is3D() {
        return H > 1;
    }

    public int volume() {
        return W * D * H;
    }

    /**
     * Checks if a box of given dimensions fits at the specified position.
     *
     * @param x x-coordinate
     * @param y y-coordinate
     * @param z z-coordinate
     * @param w width
     * @param d depth
     * @param h height
     * @return true if the box fits, false otherwise
     */
    public boolean fits(int x, int y, int z, int w, int d, int h) {
        return x + w <= W && y + d <= D && z + h <= H;
    }
}
