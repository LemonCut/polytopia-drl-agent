package players.heuristics;

import core.Types;
import core.game.GameState;

import java.util.ArrayList;

public class TribesNoHeuristic implements StateHeuristic {

    private int playerID;
    private int WIN_BOOST = 100000000;
    private int LOSE_BOOST = -100000000;

    public TribesNoHeuristic(int playerID, ArrayList<Integer> allIds) {
        this.playerID = playerID;
    }

    @Override
    public double evaluateState(GameState gameState) {
        int boost = 0;
        if(gameState.isGameOver()) {
            if(gameState.getTribeWinStatus(playerID) == Types.RESULT.WIN)
                boost = WIN_BOOST;
            else if (gameState.getTribeWinStatus(playerID) == Types.RESULT.LOSS)
                boost = LOSE_BOOST;
        }
        
        return boost + gameState.getScore(playerID);
    }

    @Override
    public double evaluateState(GameState oldState, GameState newState) {
        return evaluateState(newState);
    }
}
