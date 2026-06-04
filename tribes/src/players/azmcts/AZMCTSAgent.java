package players.azmcts;

import core.actions.Action;
import core.game.GameState;
import players.Agent;
import players.PythonBridge;
import utils.ElapsedCpuTimer;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.Serializable;
import java.util.*;

/**
 * Simple single-threaded PUCT MCTS agent that queries the Python NN for priors and value.
 * This is a conservative, self-contained implementation intended as a starting point.
 */
public class AZMCTSAgent extends Agent {

    private final Random rnd;
    private final int nSimulations;
    private final double cPuct;

    public AZMCTSAgent(long seed, int nSimulations, double cPuct) {
        super(seed);
        this.rnd = new Random(seed);
        this.nSimulations = nSimulations;
        this.cPuct = cPuct;
    }

    public AZMCTSAgent(long seed, int nSimulations) { this(seed, nSimulations, 1.5); }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        ArrayList<Action> rootActions = gs.getAllAvailableActions();
        int n = rootActions.size();
        if (n == 0) return null;
        if (n == 1) return rootActions.get(0);

        Node root = new Node(rootActions);

        // Expand root once with NN priors
        expandNodeWithNN(root, gs);

        // Run simulations
        for (int sim = 0; sim < nSimulations; sim++) {
            GameState gsSim = gs.copy();
            simulateFrom(root, gsSim);
        }

        // Choose action with highest visit count
        int bestIdx = -1;
        double bestN = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < root.N.length; i++) {
            if (root.N[i] > bestN) {
                bestN = root.N[i];
                bestIdx = i;
            }
        }
        if (bestIdx < 0) bestIdx = 0;
        return root.actions.get(bestIdx);
    }

    private void simulateFrom(Node root, GameState gsSim) {
        List<TraversalStep> path = new ArrayList<>();
        Node node = root;

        // selection
        while (true) {
            if (gsSim.isGameOver()) {
                // terminal
                double value = terminalValue(gsSim);
                backpropagate(path, value);
                return;
            }

            if (!node.expanded) {
                // leaf - expand
                double value = expandNodeWithNN(node, gsSim);
                backpropagate(path, value);
                return;
            }

            // select child
            int bestIdx = selectActionIndex(node);
            // push to path
            path.add(new TraversalStep(node, bestIdx));

            // advance state
            Action a = node.actions.get(bestIdx);
            gsSim.advance(a, true);

            // move to child node (create if absent)
            if (node.children.get(bestIdx) == null) {
                ArrayList<Action> childActions = gsSim.getAllAvailableActions();
                Node child = new Node(childActions);
                node.children.put(bestIdx, child);
            }
            node = node.children.get(bestIdx);
        }
    }

    private int selectActionIndex(Node node) {
        double sumN = 0.0;
        for (double ni : node.N) sumN += ni;
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestIdx = 0;
        for (int i = 0; i < node.actions.size(); i++) {
            double q = node.N[i] > 0 ? node.W[i] / node.N[i] : 0.0;
            double u = cPuct * node.P[i] * Math.sqrt(sumN + 1e-8) / (1.0 + node.N[i]);
            double score = q + u;
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    private double expandNodeWithNN(Node node, GameState gs) {
        try {
            String resp = PythonBridge.queryPolicy(gs, node.actions);
            JSONObject policyResponse = new JSONObject(resp);
            String status = policyResponse.optString("status", "error");
            double value = 0.0;
            JSONArray atProbs = null, srcProbs = null, tgtProbs = null, prmProbs = null;

            if ("success".equals(status)) {
                atProbs = policyResponse.optJSONArray("action_type_probs");
                srcProbs = policyResponse.optJSONArray("source_probs");
                tgtProbs = policyResponse.optJSONArray("target_probs");
                prmProbs = policyResponse.optJSONArray("param_probs");
                value = policyResponse.optDouble("value", 0.0);
            }

            int m = node.actions.size();
            node.P = new double[m];
            node.N = new double[m];
            node.W = new double[m];

            double sum = 0.0;
            for (int i = 0; i < m; i++) {
                Action action = node.actions.get(i);
                JSONObject comps = PythonBridge.encodeActionComponents(action, gs);
                double prior = jointProbFromComponents(action, comps, atProbs, srcProbs, tgtProbs, prmProbs);
                node.P[i] = prior;
                sum += prior;
                node.N[i] = 0.0;
                node.W[i] = 0.0;
            }
            // normalize priors
            if (sum > 0) {
                for (int i = 0; i < m; i++) node.P[i] /= sum;
            } else {
                double u = 1.0 / m;
                for (int i = 0; i < m; i++) node.P[i] = u;
            }

            node.expanded = true;
            node.children = new HashMap<>();
            return value;
        } catch (Exception e) {
            // NN not available, fall back to uniform priors and zero value
            int m = node.actions.size();
            node.P = new double[m];
            node.N = new double[m];
            node.W = new double[m];
            for (int i = 0; i < m; i++) {
                node.P[i] = 1.0 / m;
                node.N[i] = 0.0;
                node.W[i] = 0.0;
            }
            node.expanded = true;
            node.children = new HashMap<>();
            return 0.0;
        }
    }

    private double terminalValue(GameState gs) {
        // return +1 for win, -1 for loss for the active tribe
        // If draw or unknown, return 0
        if (!gs.isGameOver()) return 0.0;
        int active = gs.getActiveTribeID();
        try {
            // Lookup winner status for the active tribe
            if (gs.getTribe(active).getWinner() == core.Types.RESULT.WIN) return 1.0;
            if (gs.getTribe(active).getWinner() == core.Types.RESULT.LOSS) return -1.0;
        } catch (Exception ignored) {}
        return 0.0;
    }

    private void backpropagate(List<TraversalStep> path, double value) {
        // Propagate value back up the path, flipping sign at each step
        double v = value;
        for (int i = path.size() - 1; i >= 0; i--) {
            TraversalStep step = path.get(i);
            Node node = step.node;
            int idx = step.actionIndex;
            node.N[idx] += 1.0;
            node.W[idx] += v;
            // flip sign for parent
            v = -v;
        }
    }

    private double jointProbFromComponents(Action action, JSONObject comps, JSONArray atProbs, JSONArray srcProbs, JSONArray tgtProbs, JSONArray prmProbs) {
        double p = 0.0;
        if (atProbs == null) return 1.0;
        int atIdx = comps.optInt("action_type_index", 0);
        p = probAt(atProbs, atIdx);

        switch (action.getActionType()) {
            case END_TURN:
                return p;
            case MOVE:
            case ATTACK:
            case CAPTURE:
            case CONVERT:
                p *= probAt(srcProbs, comps.optInt("source_actor_index", 0));
                p *= probAt(tgtProbs, comps.optInt("target_actor_index", 0));
                return p;
            case BUILD_ROAD:
            case DECLARE_WAR:
                p *= probAt(tgtProbs, comps.optInt("target_actor_index", 0));
                return p;
            case SEND_STARS:
                p *= probAt(tgtProbs, comps.optInt("target_actor_index", 0));
                p *= probAt(prmProbs, comps.optInt("param_index", 0));
                return p;
            case RESEARCH_TECH:
                p *= probAt(prmProbs, comps.optInt("param_index", 0));
                return p;
            case BUILD:
                p *= probAt(srcProbs, comps.optInt("source_actor_index", 0));
                p *= probAt(tgtProbs, comps.optInt("target_actor_index", 0));
                p *= probAt(prmProbs, comps.optInt("param_index", 0));
                return p;
            case SPAWN:
                p *= probAt(srcProbs, comps.optInt("source_actor_index", 0));
                p *= probAt(prmProbs, comps.optInt("param_index", 0));
                return p;
            default:
                // Fallback: multiply source/target/param if present
                p *= probAt(srcProbs, comps.optInt("source_actor_index", 0));
                p *= probAt(tgtProbs, comps.optInt("target_actor_index", 0));
                p *= probAt(prmProbs, comps.optInt("param_index", 0));
                return p;
        }
    }

    private double probAt(JSONArray probs, int index) {
        if (probs == null || index < 0 || index >= probs.length()) return 0.0;
        return probs.optDouble(index, 0.0);
    }

    @Override
    public Agent copy() {
        return new AZMCTSAgent(seed, nSimulations, cPuct);
    }

    private static class Node implements Serializable {
        ArrayList<Action> actions;
        double[] P; // prior
        double[] N; // visits
        double[] W; // total value
        boolean expanded = false;
        Map<Integer, Node> children = new HashMap<>();

        Node(ArrayList<Action> actions) {
            this.actions = new ArrayList<>(actions);
        }
    }

    private static class TraversalStep {
        Node node;
        int actionIndex;

        TraversalStep(Node node, int idx) { this.node = node; this.actionIndex = idx; }
    }

}
