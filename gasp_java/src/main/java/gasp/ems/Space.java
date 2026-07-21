package gasp.ems;

import java.util.Objects;

/**
 * Rappresenta un Empty Maximal Space (EMS).
 * Utilizziamo una classe (o record) invece di array per leggibilità, 
 * mantenendo i campi primitivi per performance.
 */
public class Space {
    public final int x, y, z;
    public final int x2, y2, z2;

    public Space(int x, int y, int z, int x2, int y2, int z2) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.x2 = x2;
        this.y2 = y2;
        this.z2 = z2;
    }

    public int getW() { return x2 - x; }
    public int getD() { return y2 - y; }
    public int getH() { return z2 - z; }
    
    public long getVolume() { 
        return (long) getW() * getD() * getH(); 
    }

    /**
     * Controlla se questo spazio contiene completamente un altro spazio 'o'.
     */
    public boolean contains(Space o) {
        return (this.x <= o.x && this.y <= o.y && this.z <= o.z
             && this.x2 >= o.x2 && this.y2 >= o.y2 && this.z2 >= o.z2);
    }
    
    public boolean containsFast(int ox, int oy, int oz, int ox2, int oy2, int oz2) {
        return (this.x <= ox && this.y <= oy && this.z <= oz
             && this.x2 >= ox2 && this.y2 >= oy2 && this.z2 >= oz2);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Space space = (Space) o;
        return x == space.x && y == space.y && z == space.z &&
               x2 == space.x2 && y2 == space.y2 && z2 == space.z2;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, z, x2, y2, z2);
    }
    
    @Override
    public String toString() {
        return String.format("Space(%d,%d,%d -> %d,%d,%d)", x, y, z, x2, y2, z2);
    }
}
