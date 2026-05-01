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

public class BridgeServer {

    private GameState gameState;
    private long seed;

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

        gameState.advance(actions.get(actionIndex), true);
        return buildStateResponse();
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