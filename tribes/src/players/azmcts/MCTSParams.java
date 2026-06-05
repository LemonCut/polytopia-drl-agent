package players.azmcts;

import players.heuristics.AlgParams;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@SuppressWarnings("WeakerAccess")
public class MCTSParams extends AlgParams {

    // Parameters
    public double K = Math.sqrt(2);
    public int ROLLOUT_LENGTH = 10;//10;
    public boolean ROLOUTS_ENABLED = true;
    public boolean CAPTURE_MCTS = false;
    public boolean DEBUG_DECISIONS = false;

    // AlphaZero-style neural MCTS (PUCT)
    public boolean NEURAL_PRIORS = false;
    public boolean NEURAL_VALUE = false;
    public double CPUCT = 1.5;

    public boolean DIRICHLET_ROOT_NOISE = true;
    public double DIRICHLET_ALPHA = 0.30;
    public double DIRICHLET_EPSILON = 0.25;
    public boolean FORCE_END_TURN_IN_SEARCH = false;
    public boolean MASK_SEND_STARS = true;
    public boolean SELECT_MOST_VISITED_ACTION = false;

    public void setParameterValue(String param, Object value) {
        switch(param) {
            case "K": K = (double) value; break;
            case "ROLLOUT_LENGTH": ROLLOUT_LENGTH = (int) value; break;
            case "heuristic_method": heuristic_method = (int) value; break;
            case "NEURAL_PRIORS": NEURAL_PRIORS = (boolean) value; break;
            case "NEURAL_VALUE": NEURAL_VALUE = (boolean) value; break;
            case "CPUCT": CPUCT = (double) value; break;
            case "DIRICHLET_ROOT_NOISE": DIRICHLET_ROOT_NOISE = (boolean) value; break;
            case "DIRICHLET_ALPHA": DIRICHLET_ALPHA = (double) value; break;
            case "DIRICHLET_EPSILON": DIRICHLET_EPSILON = (double) value; break;
            case "FORCE_END_TURN_IN_SEARCH": FORCE_END_TURN_IN_SEARCH = (boolean) value; break;
            case "DEBUG_DECISIONS": DEBUG_DECISIONS = (boolean) value; break;
            case "MASK_SEND_STARS": MASK_SEND_STARS = (boolean) value; break;
            case "SELECT_MOST_VISITED_ACTION": SELECT_MOST_VISITED_ACTION = (boolean) value; break;
        }
    }

    public Object getParameterValue(String param) {
        switch(param) {
            case "K": return K;
            case "ROLLOUT_LENGTH": return ROLLOUT_LENGTH;
            case "heuristic_method": return heuristic_method;
            case "NEURAL_PRIORS": return NEURAL_PRIORS;
            case "NEURAL_VALUE": return NEURAL_VALUE;
            case "CPUCT": return CPUCT;
            case "DIRICHLET_ROOT_NOISE": return DIRICHLET_ROOT_NOISE;
            case "DIRICHLET_ALPHA": return DIRICHLET_ALPHA;
            case "DIRICHLET_EPSILON": return DIRICHLET_EPSILON;
            case "FORCE_END_TURN_IN_SEARCH": return FORCE_END_TURN_IN_SEARCH;
            case "DEBUG_DECISIONS": return DEBUG_DECISIONS;
            case "MASK_SEND_STARS": return MASK_SEND_STARS;
            case "SELECT_MOST_VISITED_ACTION": return SELECT_MOST_VISITED_ACTION;
        }
        return null;
    }


    public ArrayList<String> getParameters() {
        ArrayList<String> paramList = new ArrayList<>();
        paramList.add("K");
        paramList.add("rollout_depth");
        paramList.add("heuristic_method");
        paramList.add("NEURAL_PRIORS");
        paramList.add("NEURAL_VALUE");
        paramList.add("CPUCT");
        paramList.add("DIRICHLET_ROOT_NOISE");
        paramList.add("DIRICHLET_ALPHA");
        paramList.add("DIRICHLET_EPSILON");
        paramList.add("FORCE_END_TURN_IN_SEARCH");
        paramList.add("DEBUG_DECISIONS");
        paramList.add("MASK_SEND_STARS");
        paramList.add("SELECT_MOST_VISITED_ACTION");
        return paramList;
    }

    public Map<String, Object[]> getParameterValues() {
        HashMap<String, Object[]> parameterValues = new HashMap<>();
        parameterValues.put("K", new Double[]{1.0, Math.sqrt(2), 2.0});
        parameterValues.put("rollout_depth", new Integer[]{5, 8, 10, 12, 15});
        parameterValues.put("heuristic_method", new Integer[]{ENTROPY_HEURISTIC, SIMPLE_HEURISTIC, DIFF_HEURISTIC});
        parameterValues.put("NEURAL_PRIORS", new Boolean[]{false, true});
        parameterValues.put("NEURAL_VALUE", new Boolean[]{false, true});
        parameterValues.put("CPUCT", new Double[]{0.5, 1.0, 1.5, 2.0});
        parameterValues.put("DIRICHLET_ROOT_NOISE", new Boolean[]{false, true});
        parameterValues.put("DIRICHLET_ALPHA", new Double[]{0.03, 0.15, 0.30, 1.0});
        parameterValues.put("DIRICHLET_EPSILON", new Double[]{0.0, 0.10, 0.25});
        parameterValues.put("FORCE_END_TURN_IN_SEARCH", new Boolean[]{false, true});
        parameterValues.put("DEBUG_DECISIONS", new Boolean[]{false, true});
        parameterValues.put("MASK_SEND_STARS", new Boolean[]{false, true});
        parameterValues.put("SELECT_MOST_VISITED_ACTION", new Boolean[]{false, true});
        return parameterValues;
    }

}
