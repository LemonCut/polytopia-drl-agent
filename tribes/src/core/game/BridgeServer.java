package core.game;

import core.Types;
import core.actors.Tribe;
import core.actions.Action;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import players.Agent;
import gui.GUI;
import gui.WindowInput;
import players.KeyController;
import players.ActionController;

public class BridgeServer {

    private GameState gameState;
    private long seed;
    private Agent[] agents;
    private GUI frame;
    private Game dummyGame;
    private players.ActionController ac;

    public static void main(String[] args) throws Exception {
        new BridgeServer().run();
    }

    private void run() throws Exception {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(System.out)), true);

        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }

            JSONObject response;
            try {
                response = handle(new JSONObject(line));
            } catch (Exception e) {
                response = new JSONObject();
                response.put("ok", false);
                response.put("error", e.getClass().getSimpleName() + ": " + e.getMessage());
            }

            writer.println(response.toString());
            writer.flush();
            if (response.optBoolean("shutdown", false)) {
                break;
            }
        }
    }

    private JSONObject handle(JSONObject message) {
        String cmd = message.getString("cmd");
        switch (cmd) {
            case "reset":
                return handleReset(message);
            case "observe":
                return handleObserve();
            case "actions":
                return handleActions();
            case "step":
                return handleStep(message.getInt("actionIndex"));
            case "agent_step":
                return handleAgentStep();
            case "close":
                JSONObject closed = new JSONObject();
                closed.put("ok", true);
                closed.put("shutdown", true);
                return closed;
            default:
                throw new IllegalArgumentException("Unknown command: " + cmd);
        }
    }

    private JSONObject handleReset(JSONObject message) {
        seed = message.optLong("seed", System.currentTimeMillis());
        Types.GAME_MODE gameMode = parseGameMode(message.opt("gameMode"));

        gameState = new GameState(new Random(seed), gameMode);

        if (message.has("levelFile")) {
            gameState.init(message.getString("levelFile"));
        } else if (message.has("levelSeed")) {
            Types.TRIBE[] tribes = parseTribes(message.getJSONArray("tribes"));
            gameState.init(message.getLong("levelSeed"), tribes);
        } else {
            throw new IllegalArgumentException("reset requires either levelFile or levelSeed");
        }

        Tribe[] currentTribes = gameState.getTribes();
        agents = new Agent[currentTribes.length];
        ac = new players.ActionController();
        
        if (message.has("agents")) {
            JSONArray agentNames = message.getJSONArray("agents");
            ArrayList<Integer> allIds = new ArrayList<>();
            for (int i = 0; i < currentTribes.length; i++) allIds.add(i);
            
            for (int i = 0; i < agentNames.length(); i++) {
                String agentName = agentNames.getString(i);
                if (agentName.equalsIgnoreCase("Python")) {
                    agents[i] = new players.PythonAgent(seed);
                } else {
                    agents[i] = createJavaAgent(agentName, i, allIds, seed, ac);
                }
            }
        }

        if (message.optBoolean("visuals", false)) {
            core.Constants.VISUALS = true;
            dummyGame = new Game();
            dummyGame.initForBridge(gameState, agents);
            
            WindowInput wi = new WindowInput();
            wi.windowClosed = false;
            players.KeyController ki = new players.KeyController(true);
            
            frame = new GUI(dummyGame, "Tribes - Bridge", ki, wi, ac, false);
            frame.addWindowListener(wi);
            frame.addKeyListener(ki);
            frame.update(gameState, null);
            frame.update(gameState, null);
        }

        prepareTurnStart();
        return buildStateResponse();
    }

    private JSONObject handleObserve() {
        ensureGameState();
        return buildStateResponse();
    }

    private JSONObject handleActions() {
        ensureGameState();
        JSONObject response = baseResponse();
        response.put("actions", serializeActions(gameState.getAllAvailableActions()));
        return response;
    }

    private JSONObject handleStep(int actionIndex) {
        ensureGameState();
        ArrayList<Action> actions = gameState.getAllAvailableActions();
        if (actionIndex < 0 || actionIndex >= actions.size()) {
            throw new IllegalArgumentException("Invalid actionIndex: " + actionIndex);
        }

        Action action = actions.get(actionIndex);
        
        if (frame != null) {
            frame.update(gameState, action);
            waitGui();
        }
        
        advanceGameState(action);
        
        if (frame != null) {
            frame.update(gameState, null);
        }
        
        return buildStateResponse();
    }

    private JSONObject handleAgentStep() {
        ensureGameState();
        int activeTribe = gameState.getActiveTribeID();
        Agent ag = agents[activeTribe];
        if (ag == null) {
            throw new IllegalStateException("Active tribe " + activeTribe + " is not a Java agent");
        }
        utils.ElapsedCpuTimer ect = new utils.ElapsedCpuTimer();
        ect.setMaxTimeMillis(core.Constants.TURN_TIME_MILLIS);
        Action action = ag.act(gameState.copy(activeTribe), ect);
        
        while (action == null && ag instanceof players.HumanAgent) {
            if (frame != null) frame.repaint();
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {}
            action = ag.act(gameState.copy(activeTribe), ect);
        }
        
        if (frame != null) {
            frame.update(gameState, action);
            waitGui();
        }
        
        advanceGameState(action);
        
        if (frame != null) {
            frame.update(gameState, null);
        }
        
        return buildStateResponse();
    }
    
    private void waitGui() {
        if (dummyGame != null && frame != null) {
            while (dummyGame.isAnimationPaused()) {
                frame.repaint();
                try {
                    Thread.sleep(10);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void advanceGameState(Action action) {
        if (action == null) return;
        int oldTribeId = gameState.getActiveTribeID();
        gameState.advance(action, true);
        int newTribeId = gameState.getActiveTribeID();
        
        if (action.getActionType() == core.Types.ACTION.END_TURN) {
            if (newTribeId <= oldTribeId) {
                gameState.incTick();
            }
        }
    }

    private void prepareTurnStart() {
        if (gameState.getBoard().getActiveTribeID() < 0) {
            gameState.getBoard().setActiveTribeID(0);
        }
        Tribe activeTribe = gameState.getActiveTribe();
        gameState.initTurn(activeTribe);
        gameState.computePlayerActions(activeTribe);
    }

    private JSONObject buildStateResponse() {
        JSONObject response = baseResponse();
        response.put("state", GameSnapshot.build(gameState, seed));
        response.put("actions", serializeActions(gameState.getAllAvailableActions()));
        return response;
    }

    private JSONObject baseResponse() {
        JSONObject response = new JSONObject();
        response.put("ok", true);
        return response;
    }

    private JSONArray serializeActions(ArrayList<Action> actions) {
        JSONArray jsonActions = new JSONArray();
        for (int i = 0; i < actions.size(); i++) {
            Action action = actions.get(i);
            JSONObject obj = new JSONObject();
            obj.put("index", i);
            obj.put("type", action.getActionType().name());
            obj.put("text", action.toString());
            jsonActions.put(obj);
        }
        return jsonActions;
    }

    private void ensureGameState() {
        if (gameState == null) {
            throw new IllegalStateException("Bridge has not been reset yet");
        }
    }

    private Agent createJavaAgent(String agentName, int playerID, ArrayList<Integer> allIds, long seed, players.ActionController ac) {
        Agent ag = null;
        if (agentName.equalsIgnoreCase("MCTS")) {
            players.mcts.MCTSParams params = new players.mcts.MCTSParams();
            params.stop_type = params.STOP_FMCALLS;
            params.heuristic_method = params.DIFF_HEURISTIC;
            params.PRIORITIZE_ROOT = true;
            params.ROLLOUT_LENGTH = 10;
            params.FORCE_TURN_END = 11;
            params.ROLOUTS_ENABLED = true;
            ag = new players.mcts.MCTSPlayer(seed, params);
        } else if (agentName.equalsIgnoreCase("PURE_MCTS")) {
            players.mcts.MCTSParams params = new players.mcts.MCTSParams();
            params.stop_type = params.STOP_FMCALLS;
            params.heuristic_method = params.NO_HEURISTIC;
            params.PRIORITIZE_ROOT = false;
            params.ROLLOUT_LENGTH = 10;
            params.FORCE_TURN_END = 11;
            params.ROLOUTS_ENABLED = true;
            ag = new players.mcts.MCTSPlayer(seed, params);
        } else if (agentName.equalsIgnoreCase("RHEA")) {
            players.rhea.RHEAParams params = new players.rhea.RHEAParams();
            params.stop_type = params.STOP_FMCALLS;
            params.heuristic_method = params.DIFF_HEURISTIC;
            params.INDIVIDUAL_LENGTH = 10;
            params.FORCE_TURN_END = 11;
            params.POP_SIZE = 10;
            ag = new players.rhea.RHEAAgent(seed, params);
        } else if (agentName.equalsIgnoreCase("OSLA")) {
            players.osla.OSLAParams params = new players.osla.OSLAParams();
            params.stop_type = params.STOP_FMCALLS;
            params.heuristic_method = params.DIFF_HEURISTIC;
            ag = new players.osla.OneStepLookAheadAgent(seed, params);
        } else if (agentName.equalsIgnoreCase("Random")) {
            ag = new players.RandomAgent(seed);
        } else if (agentName.equalsIgnoreCase("DoNothing")) {
            ag = new players.DoNothingAgent(seed);
        } else if (agentName.equalsIgnoreCase("RuleBased") || agentName.equalsIgnoreCase("Simple")) {
            ag = new players.SimpleAgent(seed);
        } else if (agentName.equalsIgnoreCase("Human")) {
            ag = new players.HumanAgent(ac);
        } else if (agentName.equalsIgnoreCase("NeuralPolicyAgent")) {
            ag = new players.NeuralPolicyAgent(seed);
        } else if (agentName.equalsIgnoreCase("AZMCTSAgent") || agentName.equalsIgnoreCase("AZ_MCTS")) {
            players.azmcts.MCTSParams params = new players.azmcts.MCTSParams();
            params.stop_type = params.STOP_ITERATIONS;
            params.num_iterations = 128;
            params.heuristic_method = params.DIFF_HEURISTIC;
            params.PRIORITIZE_ROOT = false; // AZ usually evaluates all actions together
            params.ROLLOUT_LENGTH = 10;
            params.FORCE_TURN_END = 11;
            params.ROLOUTS_ENABLED = true;
            params.NEURAL_PRIORS = true;
            params.NEURAL_VALUE = true;
            ag = new players.azmcts.MCTSPlayer(seed, params);
        } else {
            throw new IllegalArgumentException("Unsupported Java agent type: " + agentName);
        }
        ag.setPlayerIDs(playerID, allIds);
        return ag;
    }

    private Types.GAME_MODE parseGameMode(Object rawValue) {
        if (rawValue == null) {
            return Types.GAME_MODE.SCORE;
        }
        if (rawValue instanceof Number) {
            return Types.GAME_MODE.getTypeByKey(((Number) rawValue).intValue());
        }
        String value = rawValue.toString();
        if (value.equalsIgnoreCase("CAPITALS")) {
            return Types.GAME_MODE.CAPITALS;
        }
        if (value.equalsIgnoreCase("SCORE")) {
            return Types.GAME_MODE.SCORE;
        }
        return Types.GAME_MODE.getTypeByKey(Integer.parseInt(value));
    }

    private Types.TRIBE[] parseTribes(JSONArray values) {
        Types.TRIBE[] tribes = new Types.TRIBE[values.length()];
        for (int i = 0; i < values.length(); i++) {
            Object rawValue = values.get(i);
            if (rawValue instanceof Number) {
                tribes[i] = Types.TRIBE.getTypeByKey(((Number) rawValue).intValue());
            } else {
                tribes[i] = parseTribeName(rawValue.toString());
            }
        }
        return tribes;
    }

    private Types.TRIBE parseTribeName(String value) {
        switch (value) {
            case "Xin Xi":
            case "Xin-Xi":
                return Types.TRIBE.XIN_XI;
            case "Imperius":
                return Types.TRIBE.IMPERIUS;
            case "Bardur":
                return Types.TRIBE.BARDUR;
            case "Oumaji":
                return Types.TRIBE.OUMAJI;
            case "Kickoo":
                return Types.TRIBE.KICKOO;
            case "Hoodrick":
                return Types.TRIBE.HOODRICK;
            case "Luxidoor":
                return Types.TRIBE.LUXIDOOR;
            case "Vengir":
                return Types.TRIBE.VENGIR;
            case "Zebasi":
                return Types.TRIBE.ZEBASI;
            case "Ai-Mo":
                return Types.TRIBE.AI_MO;
            case "Quetzali":
                return Types.TRIBE.QUETZALI;
            case "Yadakk":
                return Types.TRIBE.YADAKK;
            default:
                throw new IllegalArgumentException("Unknown tribe: " + value);
        }
    }
}