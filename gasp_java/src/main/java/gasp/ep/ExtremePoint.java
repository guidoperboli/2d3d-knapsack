package gasp.ep;

import java.util.Objects;

/**
 * Extreme Point representation with its coordinates and Residual Space (RS).
 * The RS fields are mutable as they are updated when new items are added.
 */
public class ExtremePoint {
    private final int x;
    private final int y;
    private final int z;
    
    private int rsX;
    private int rsY;
    private int rsZ;

    public ExtremePoint(int x, int y, int z, int rsX, int rsY, int rsZ) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.rsX = rsX;
        this.rsY = rsY;
        this.rsZ = rsZ;
    }

    public int getX() { return x; }
    public int getY() { return y; }
    public int getZ() { return z; }
    
    public int getRsX() { return rsX; }
    public void setRsX(int rsX) { this.rsX = rsX; }
    
    public int getRsY() { return rsY; }
    public void setRsY(int rsY) { this.rsY = rsY; }
    
    public int getRsZ() { return rsZ; }
    public void setRsZ(int rsZ) { this.rsZ = rsZ; }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ExtremePoint that = (ExtremePoint) o;
        return x == that.x && y == that.y && z == that.z;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, z);
    }
}
