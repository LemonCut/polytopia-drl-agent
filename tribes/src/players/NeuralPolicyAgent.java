package players;

import core.Types;
import core.actions.Action;
import core.game.GameState;
import org.json.JSONObject;
import utils.ElapsedCpuTimer;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * Policy-only neural agent used for fast arena evaluation.
 * It samples directly from the network's legal-action policy, without MCTS or
 * Dirichlet noise.
 */
public class NeuralPolicyAgent extends Agent {

    private final Random rnd;

    public NeuralPolicyAgent(long seed) {
        super(seed);
        this.rnd = new Random(seed);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        ArrayList<Action> allActions = gs.getAllAvailableActions();
        if (allActions == null || allActions.isEmpty()) {
            return null;
        }
        if (allActions.size() == 1) {
            return allActions.get(0);
        }

        try {
            JSONObject response = PythonBridge.queryPolicyJson(gs, allActions, policyUrl());
            if ("success".equals(response.optString("status", "error"))) {
                double[] priors = PythonBridge.actionPriorsFromPolicy(allActions, gs, response);
                int selected = sampleFromDistribution(priors);
                if (selected >= 0 && selected < allActions.size()) {
                    return allActions.get(selected);
                }
            }
        } catch (Exception e) {
            System.out.println("[NeuralPolicyAgent] policy query failed: " + e.getMessage());
        }

        return allActions.get(rnd.nextInt(allActions.size()));
    }

    @Override
    public void result(GameState gs, double reward) {
        String resultFile = System.getenv("TRIBES_EVAL_RESULT_FILE");
        if (resultFile == null || resultFile.isEmpty()) {
            return;
        }

        Types.RESULT winner = Types.RESULT.INCOMPLETE;
        try {
            winner = gs.getTribe(playerID).getWinner();
        } catch (Exception ignored) {
            // keep default
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
            System.out.println("[NeuralPolicyAgent] could not write eval result: " + e.getMessage());
        }
    }

    @Override
    public Agent copy() {
        return new NeuralPolicyAgent(seed);
    }

    private String policyUrl() {
        return PythonBridge.policyUrl(playerID);
    }

    private int sampleFromDistribution(double[] probs) {
        if (probs == null || probs.length == 0) {
            return -1;
        }

        double sum = 0.0;
        for (double prob : probs) {
            if (Double.isFinite(prob) && prob > 0.0) {
                sum += prob;
            }
        }

        if (sum <= 0.0) {
            return rnd.nextInt(probs.length);
        }

        double r = rnd.nextDouble() * sum;
        double c = 0.0;
        for (int i = 0; i < probs.length; i++) {
            double p = probs[i];
            if (!Double.isFinite(p) || p <= 0.0) {
                continue;
            }
            c += p;
            if (r <= c) {
                return i;
            }
        }
        return probs.length - 1;
    }
}
