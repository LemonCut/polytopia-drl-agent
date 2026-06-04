package players;

import core.actions.Action;
import core.game.GameState;
import utils.ElapsedCpuTimer;

public class PythonAgent extends Agent {
    public PythonAgent(long seed) {
        super(seed);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        throw new UnsupportedOperationException("PythonAgent act() should never be called in Java");
    }
    
    @Override
    public Agent copy() {
        return new PythonAgent(seed);
    }
}
