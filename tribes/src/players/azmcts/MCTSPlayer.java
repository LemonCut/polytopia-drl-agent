package players.azmcts;

import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.actions.tribeactions.SendStars;
import core.actors.Tribe;
import core.game.GameState;
import core.Types;
import players.Agent;
import players.PythonBridge;
import utils.ElapsedCpuTimer;
import org.json.JSONObject;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.Random;

public class MCTSPlayer extends Agent {

    private Random m_rnd;
    private MCTSParams params;

    public MCTSPlayer(long seed)
    {
        super(seed);
        m_rnd = new Random(seed);
        this.params = new MCTSParams();
    }

    public MCTSPlayer(long seed, MCTSParams params) {
        this(seed);
        this.params = params;
    }

    public Action act(GameState gs, ElapsedCpuTimer ect) {
        //Gather all available actions:
        ArrayList<Action> allActions = gs.getAllAvailableActions();

        if(allActions.size() == 1)
            return allActions.get(0); //EndTurn, it's possible.

        ArrayList<Action> rootActions = params.PRIORITIZE_ROOT ? determineActionGroup(gs, m_rnd) : allActions;
        if(rootActions == null)
            return new EndTurn();
        rootActions = maskSendStars(rootActions);

        SingleTreeNode m_root = new SingleTreeNode(params, m_rnd, rootActions.size(), rootActions, this.playerID);
        m_root.setRootGameState(m_root, gs, allPlayerIDs);

        m_root.mctsSearch(ect);

        if (params.CAPTURE_MCTS) {
            int[] rootVisits = m_root.getVisitCounts();
            int[] visitCounts = alignVisitCounts(allActions, rootActions, rootVisits);
            PythonBridge.captureMctsSample(gs, allActions, visitCounts, m_root.getRootValue(), playerID);
        }
        int selectedAction = params.SELECT_MOST_VISITED_ACTION
                ? m_root.mostVisitedAction()
                : m_root.sampleVisitCountAction();
        Action chosen = rootActions.get(selectedAction);

        if (params.DEBUG_DECISIONS) {
            System.out.println("\n===== MCTS DECISION =====");
            System.out.println("Player: " + playerID);
            System.out.println("Available actions: " + rootActions.size());
            System.out.println("Tick: " + gs.getTick());
            System.out.println("Chosen action: " + chosen);

            int[] rootVisits = m_root.getVisitCounts();
            int[] visitCounts = alignVisitCounts(allActions, rootActions, rootVisits);

            System.out.println("---- Visit counts ----");
            for (int i = 0; i < allActions.size(); i++) {
                Action a = allActions.get(i);
                int v = visitCounts[i];

                // highlight chosen action
                boolean isChosen = a == chosen;

                System.out.println(
                    (isChosen ? ">> " : "   ") +
                    "[" + i + "] " + a +
                    " | visits=" + v +
                    actionDeltaSummary(gs, a)
                );
            }

            System.out.println("========================\n");
        }

        return chosen;


    }

    private String actionDeltaSummary(GameState gs, Action action) {
        try {
            int activeTribeID = gs.getActiveTribeID();
            Tribe beforeActive = gs.getTribe(activeTribeID);
            int beforeActiveScore = beforeActive.getScore();
            int beforeActiveStars = beforeActive.getStars();

            Integer targetTribeID = null;
            int beforeTargetScore = 0;
            int beforeTargetStars = 0;
            if (action instanceof SendStars) {
                targetTribeID = ((SendStars) action).getTargetID();
                Tribe beforeTarget = gs.getTribe(targetTribeID);
                beforeTargetScore = beforeTarget.getScore();
                beforeTargetStars = beforeTarget.getStars();
            }

            GameState next = gs.copy();
            next.advance(action, false);

            Tribe afterActive = next.getTribe(activeTribeID);
            String summary = " | p" + activeTribeID
                    + " score " + beforeActiveScore + "->" + afterActive.getScore()
                    + ", stars " + beforeActiveStars + "->" + afterActive.getStars();

            if (targetTribeID != null) {
                Tribe afterTarget = next.getTribe(targetTribeID);
                summary += " | target p" + targetTribeID
                        + " score " + beforeTargetScore + "->" + afterTarget.getScore()
                        + ", stars " + beforeTargetStars + "->" + afterTarget.getStars();
            }

            return summary;
        } catch (Exception e) {
            return " | delta unavailable: " + e.getMessage();
        }
    }


    @Override
    public Agent copy() {
        return null;
    }

    @Override
    public void result(GameState gs, double reward) {
        Types.RESULT winner = Types.RESULT.INCOMPLETE;
        try {
            winner = gs.getTribe(playerID).getWinner();
        } catch (Exception ignored) {
            // keep default
        }

        writeEvalResult(gs, reward, winner);

        if (params.CAPTURE_MCTS) {
            PythonBridge.reportGameResult(gs, playerID, reward, winner);
        }
    }

    private void writeEvalResult(GameState gs, double reward, Types.RESULT winner) {
        String resultFile = System.getenv("TRIBES_EVAL_RESULT_FILE");
        if (resultFile == null || resultFile.isEmpty()) {
            return;
        }

        JSONObject payload = new JSONObject();
        payload.put("player_id", playerID);
        payload.put("winner", winner.name());
        payload.put("reward", reward);
        payload.put("game_seed", -1);
        payload.put("tick", gs.getTick());

        try (FileWriter writer = new FileWriter(resultFile, true)) {
            writer.write(payload.toString());
            writer.write(System.lineSeparator());
        } catch (IOException e) {
            System.out.println("[MCTSPlayer] could not write eval result: " + e.getMessage());
        }
    }

    private int[] alignVisitCounts(ArrayList<Action> allActions, ArrayList<Action> rootActions, int[] rootVisits) {
        int[] aligned = new int[allActions.size()];
        if (rootActions == allActions) {
            System.arraycopy(rootVisits, 0, aligned, 0, Math.min(rootVisits.length, aligned.length));
            return aligned;
        }

        IdentityHashMap<Action, Integer> indexByAction = new IdentityHashMap<>();
        for (int i = 0; i < allActions.size(); i++) {
            indexByAction.put(allActions.get(i), i);
        }

        int limit = Math.min(rootActions.size(), rootVisits.length);
        for (int i = 0; i < limit; i++) {
            Integer idx = indexByAction.get(rootActions.get(i));
            if (idx != null) {
                aligned[idx] = rootVisits[i];
            }
        }

        return aligned;
    }

    private ArrayList<Action> maskSendStars(ArrayList<Action> actions) {
        if (!params.MASK_SEND_STARS || actions == null) {
            return actions;
        }

        // This is intentionally an action-space mask, not just a neural-logit
        // mask. If SendStars stayed in the Java action list, root Dirichlet
        // noise could still give it search mass.
        // Set TRIBES_MASK_SEND_STARS=false to evaluate/train with SendStars.
        ArrayList<Action> filtered = new ArrayList<>();
        for (Action action : actions) {
            if (!(action instanceof SendStars)) {
                filtered.add(action);
            }
        }

        // The game normally always has EndTurn, but keep a fallback so MCTS
        // remains valid if a custom state ever exposes only SendStars actions.
        return filtered.isEmpty() ? actions : filtered;
    }

}
