package gasp.geometry;

/**
 * An item accommodated in the knapsack at position (x, y, z) with a
 * specific orientation (w, d, h).
 * <p>
 * Implemented as a record for immutability and memory efficiency.
 */
public record Placement(Item item, int x, int y, int z, int w, int d, int h) {

    public int x2() {
        return x + w;
    }

    public int y2() {
        return y + d;
    }

    public int z2() {
        return z + h;
    }

    /**
     * Checks if this placement overlaps with another placement.
     *
     * @param other the other placement
     * @return true if there is an overlap, false otherwise
     */
    public boolean overlaps(Placement other) {
        return !(this.x2() <= other.x() || other.x2() <= this.x()
                || this.y2() <= other.y() || other.y2() <= this.y()
                || this.z2() <= other.z() || other.z2() <= this.z());
    }
}
