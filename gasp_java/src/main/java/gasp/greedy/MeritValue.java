package gasp.greedy;

/**
 * Represents the evaluation of an Extreme Point for a specific placement.
 * <p>
 * Implements Comparable for lexicographic sorting based on:
 * 1. The main criterion score 'f' (lower is better)
 * 2. z-coordinate (lower is better)
 * 3. y-coordinate (lower is better)
 * 4. x-coordinate (lower is better)
 *
 * @param f the primary merit score
 * @param z tie-break z
 * @param y tie-break y
 * @param x tie-break x
 */
public record MeritValue(long f, int z, int y, int x) implements Comparable<MeritValue> {

    @Override
    public int compareTo(MeritValue o) {
        if (this.f != o.f) {
            return Long.compare(this.f, o.f);
        }
        if (this.z != o.z) {
            return Integer.compare(this.z, o.z);
        }
        if (this.y != o.y) {
            return Integer.compare(this.y, o.y);
        }
        return Integer.compare(this.x, o.x);
    }
}
