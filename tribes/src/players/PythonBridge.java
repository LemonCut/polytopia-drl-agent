package players;

import core.Constants;
import core.TechnologyTree;
import core.Types;
import core.actions.Action;
import core.actors.Building;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class PythonBridge {

    private static final String DEFAULT_URL = "http://127.0.0.1:8000/query";
    private static final String CAPTURE_URL = "http://127.0.0.1:8000/capture";
    private static final String RESULT_URL = "http://127.0.0.1:8000/result";
    private static JSONObject cachedSchema = null;

    public static String queryPolicy(GameState gs, ArrayList<Action> allActions) throws IOException {
        JSONObject payload = buildPayload(gs, allActions);
        return postJson(payload.toString());
    }

    public static String queryPolicy(GameState gs, ArrayList<Action> allActions, String urlString) throws IOException {
        JSONObject payload = buildPayload(gs, allActions);
        return postJson(payload.toString(), urlString);
    }

    public static JSONObject queryPolicyJson(GameState gs, ArrayList<Action> allActions) throws IOException {
        return new JSONObject(queryPolicy(gs, allActions));
    }

    public static JSONObject queryPolicyJson(GameState gs, ArrayList<Action> allActions, String urlString) throws IOException {
        return new JSONObject(queryPolicy(gs, allActions, urlString));
    }

    public static String policyUrl(int playerID) {
        String playerSpecific = System.getenv("TRIBES_POLICY_URL_PLAYER_" + playerID);
        if (playerSpecific != null && !playerSpecific.isEmpty()) {
            return playerSpecific;
        }
        String shared = System.getenv("TRIBES_POLICY_URL");
        if (shared != null && !shared.isEmpty()) {
            return shared;
        }
        return DEFAULT_URL;
    }

    /**
     * Compute a normalized joint prior over the provided available actions.
     * Prefer composing raw component logits and normalizing over the actual legal actions.
     * Multiplying already-normalized component probabilities structurally favors actions
     * with fewer components (especially END_TURN) at random initialization.
     */
    public static double[] actionPriorsFromPolicy(ArrayList<Action> allActions, GameState gs, JSONObject policyResponse) {
        double[] priors = new double[allActions.size()];

        JSONArray actionTypeLogits = policyResponse.optJSONArray("action_type_logits");
        JSONArray sourceLogits = policyResponse.optJSONArray("source_logits");
        JSONArray targetLogits = policyResponse.optJSONArray("target_logits");
        JSONArray paramLogits = policyResponse.optJSONArray("param_logits");

        if (actionTypeLogits != null && sourceLogits != null && targetLogits != null && paramLogits != null) {
            return actionPriorsFromLogits(allActions, gs, actionTypeLogits, sourceLogits, targetLogits, paramLogits);
        }

        JSONArray actionTypeProbs = policyResponse.optJSONArray("action_type_probs");
        JSONArray sourceProbs = policyResponse.optJSONArray("source_probs");
        JSONArray targetProbs = policyResponse.optJSONArray("target_probs");
        JSONArray paramProbs = policyResponse.optJSONArray("param_probs");

        if (actionTypeProbs == null || sourceProbs == null || targetProbs == null || paramProbs == null) {
            return uniform(priors.length);
        }

        double sum = 0.0;
        for (int i = 0; i < allActions.size(); i++) {
            Action action = allActions.get(i);
            JSONObject components = encodeActionComponents(action, gs);
            double p = jointProbability(action, components, actionTypeProbs, sourceProbs, targetProbs, paramProbs);
            priors[i] = p;
            sum += p;
        }

        if (sum <= 0.0) {
            return uniform(priors.length);
        }

        for (int i = 0; i < priors.length; i++) {
            priors[i] = priors[i] / sum;
        }
        return priors;
    }

    private static double[] actionPriorsFromLogits(ArrayList<Action> allActions, GameState gs, JSONArray actionTypeLogits,
                                                   JSONArray sourceLogits, JSONArray targetLogits, JSONArray paramLogits) {
        double[] scores = new double[allActions.size()];
        if (scores.length == 0) {
            return scores;
        }

        for (int i = 0; i < allActions.size(); i++) {
            Action action = allActions.get(i);
            JSONObject components = encodeActionComponents(action, gs);
            scores[i] = jointLogitScore(action, components, actionTypeLogits, sourceLogits, targetLogits, paramLogits);
        }

        return softmax(scores);
    }

    private static double jointLogitScore(Action action, JSONObject components, JSONArray actionTypeLogits,
                                          JSONArray sourceLogits, JSONArray targetLogits, JSONArray paramLogits) {
        if (action == null) {
            return 0.0;
        }
        Types.ACTION type = action.getActionType();
        double score = probAt(actionTypeLogits, components.optInt("action_type_index", 0));
        if (type == null) {
            return score;
        }

        switch (type) {
            case END_TURN:
                return score;

            case MOVE:
            case ATTACK:
            case CAPTURE:
            case CONVERT:
                score += probAt(sourceLogits, components.optInt("source_actor_index", 0));
                score += probAt(targetLogits, components.optInt("target_actor_index", 0));
                return score;

            case BUILD_ROAD:
            case DECLARE_WAR:
                score += probAt(targetLogits, components.optInt("target_actor_index", 0));
                return score;

            case SEND_STARS:
                score += probAt(targetLogits, components.optInt("target_actor_index", 0));
                score += probAt(paramLogits, components.optInt("param_index", 0));
                return score;

            case RESEARCH_TECH:
                score += probAt(paramLogits, components.optInt("param_index", 0));
                return score;

            case BUILD:
                score += probAt(sourceLogits, components.optInt("source_actor_index", 0));
                score += probAt(targetLogits, components.optInt("target_actor_index", 0));
                score += probAt(paramLogits, components.optInt("param_index", 0));
                return score;

            case SPAWN:
                score += probAt(sourceLogits, components.optInt("source_actor_index", 0));
                score += probAt(paramLogits, components.optInt("param_index", 0));
                return score;

            case BURN_FOREST:
            case CLEAR_FOREST:
            case DESTROY:
            case GROW_FOREST:
                score += probAt(sourceLogits, components.optInt("source_actor_index", 0));
                score += probAt(targetLogits, components.optInt("target_actor_index", 0));
                return score;

            case LEVEL_UP:
                score += probAt(sourceLogits, components.optInt("source_actor_index", 0));
                score += probAt(paramLogits, components.optInt("param_index", 0));
                return score;

            case RESOURCE_GATHERING:
                score += probAt(sourceLogits, components.optInt("source_actor_index", 0));
                return score;

            case DISBAND:
            case EXAMINE:
            case HEAL_OTHERS:
            case MAKE_VETERAN:
            case RECOVER:
            case CLIMB_MOUNTAIN:
            case UPGRADE_BOAT:
            case UPGRADE_SHIP:
                score += probAt(sourceLogits, components.optInt("source_actor_index", 0));
                return score;

            default:
                return score;
        }
    }

    private static double[] softmax(double[] scores) {
        double[] out = new double[scores.length];
        if (scores.length == 0) {
            return out;
        }

        double max = -Double.MAX_VALUE;
        for (double score : scores) {
            if (Double.isFinite(score) && score > max) {
                max = score;
            }
        }
        if (max == -Double.MAX_VALUE) {
            return uniform(scores.length);
        }

        double sum = 0.0;
        for (int i = 0; i < scores.length; i++) {
            out[i] = Math.exp(scores[i] - max);
            sum += out[i];
        }
        if (sum <= 0.0 || !Double.isFinite(sum)) {
            return uniform(scores.length);
        }
        for (int i = 0; i < out.length; i++) {
            out[i] /= sum;
        }
        return out;
    }

    private static double jointProbability(Action action, JSONObject components, JSONArray actionTypeProbs, JSONArray sourceProbs, JSONArray targetProbs, JSONArray paramProbs) {
        if (action == null) {
            return 0.0;
        }
        Types.ACTION type = action.getActionType();
        double probability = probAt(actionTypeProbs, components.optInt("action_type_index", 0));
        if (type == null) {
            return probability;
        }

        switch (type) {
            case END_TURN:
                return probability;

            case MOVE:
            case ATTACK:
            case CAPTURE:
            case CONVERT:
                probability *= probAt(sourceProbs, components.optInt("source_actor_index", 0));
                probability *= probAt(targetProbs, components.optInt("target_actor_index", 0));
                return probability;

            case BUILD_ROAD:
            case DECLARE_WAR:
                probability *= probAt(targetProbs, components.optInt("target_actor_index", 0));
                return probability;

            case SEND_STARS:
                probability *= probAt(targetProbs, components.optInt("target_actor_index", 0));
                probability *= probAt(paramProbs, components.optInt("param_index", 0));
                return probability;

            case RESEARCH_TECH:
                probability *= probAt(paramProbs, components.optInt("param_index", 0));
                return probability;

            case BUILD:
                probability *= probAt(sourceProbs, components.optInt("source_actor_index", 0));
                probability *= probAt(targetProbs, components.optInt("target_actor_index", 0));
                probability *= probAt(paramProbs, components.optInt("param_index", 0));
                return probability;

            case SPAWN:
                probability *= probAt(sourceProbs, components.optInt("source_actor_index", 0));
                probability *= probAt(paramProbs, components.optInt("param_index", 0));
                return probability;

            case BURN_FOREST:
            case CLEAR_FOREST:
            case DESTROY:
            case GROW_FOREST:
                probability *= probAt(sourceProbs, components.optInt("source_actor_index", 0));
                probability *= probAt(targetProbs, components.optInt("target_actor_index", 0));
                return probability;

            case LEVEL_UP:
                probability *= probAt(sourceProbs, components.optInt("source_actor_index", 0));
                probability *= probAt(paramProbs, components.optInt("param_index", 0));
                return probability;

            case RESOURCE_GATHERING:
                probability *= probAt(sourceProbs, components.optInt("source_actor_index", 0));
                return probability;

            case DISBAND:
            case EXAMINE:
            case HEAL_OTHERS:
            case MAKE_VETERAN:
            case RECOVER:
            case CLIMB_MOUNTAIN:
            case UPGRADE_BOAT:
            case UPGRADE_SHIP:
                probability *= probAt(sourceProbs, components.optInt("source_actor_index", 0));
                return probability;

            default:
                return probability;
        }
    }

    private static double probAt(JSONArray probs, int index) {
        if (probs == null || index < 0 || index >= probs.length()) {
            return 0.0;
        }
        return probs.optDouble(index, 0.0);
    }

    private static double[] uniform(int n) {
        double[] out = new double[n];
        if (n <= 0) {
            return out;
        }
        double v = 1.0 / n;
        for (int i = 0; i < n; i++) {
            out[i] = v;
        }
        return out;
    }

    public static void captureMctsSample(GameState gs, ArrayList<Action> allActions, int[] visitCounts, double rootValue, int playerId) {
        try {
            JSONObject payload = buildPayload(gs, allActions);
            payload.put("policy_type", "mcts");
            payload.put("game_seed", -1);
            payload.put("player_id", playerId);

            JSONObject mcts = new JSONObject();
            mcts.put("visit_counts", toIntArray(visitCounts));
            mcts.put("root_value", rootValue);
            payload.put("mcts", mcts);

            postJson(payload.toString(), CAPTURE_URL);
        } catch (Exception e) {
            System.out.println("[PythonBridge] capture error: " + e.getMessage());
        }
    }

    public static void reportGameResult(GameState gs, int playerId, double reward, Types.RESULT winner) {
        try {
            JSONObject payload = new JSONObject();
            payload.put("game_seed", -1);
            payload.put("player_id", playerId);
            payload.put("reward", reward);
            payload.put("winner", winner == null ? null : winner.name());
            payload.put("tick", gs.getTick());
            postJson(payload.toString(), RESULT_URL);
        } catch (Exception e) {
            System.out.println("[PythonBridge] result error: " + e.getMessage());
        }
    }

    private static String postJson(String jsonPayload) throws IOException {
        return postJson(jsonPayload, DEFAULT_URL);
    }

    private static String postJson(String jsonPayload, String urlString) throws IOException {
        URL url = new URL(urlString);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        try {
            conn.setConnectTimeout(5000);
            conn.setReadTimeout(30000);
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json; utf-8");
            conn.setRequestProperty("Accept", "application/json");
            conn.setDoOutput(true);

            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = jsonPayload.getBytes("utf-8");
                os.write(input, 0, input.length);
            }

            int code = conn.getResponseCode();
            InputStream responseStream = (code >= 200 && code < 300)
                    ? conn.getInputStream()
                    : conn.getErrorStream();
            if (responseStream == null) {
                return "";
            }

            StringBuilder resp = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(responseStream, "utf-8"))) {
                String responseLine;
                while ((responseLine = br.readLine()) != null) {
                    resp.append(responseLine.trim());
                }
            }
            return resp.toString();
        } finally {
            conn.disconnect();
        }
    }

    private static JSONObject buildPayload(GameState gs, ArrayList<Action> allActions) {
        JSONObject payload = new JSONObject();
        Board board = gs.getBoard();
        
        Tribe activeTribe = gs.getActiveTribe();
        boolean fullObservability = Constants.PLAY_WITH_FULL_OBS;
        boolean[][] obsGrid = fullObservability || activeTribe == null ? null : activeTribe.getObsGrid();
        int activeTribeID = gs.getActiveTribeID();

        payload.put("schema_version", 1);
        payload.put("full_observability", fullObservability);
        payload.put("observability", fullObservability ? "FULL" : "PARTIAL");
        payload.put("tick", gs.getTick());
        payload.put("active_tribe_id", activeTribeID);
        payload.put("game_mode", gs.getGameMode().name());
        payload.put("is_game_over", gs.isGameOver());
        payload.put("available_action_count", allActions.size());
        payload.put("available_actions", serializeActions(allActions, gs));
        payload.put("visibility", serializeVisibility(obsGrid, board.getSize()));
        payload.put("board", serializeBoard(board, obsGrid, activeTribeID));
        payload.put("tribes", serializeTribes(board, obsGrid, activeTribeID, fullObservability));

        return payload;
    }

    private static JSONObject serializeBoard(Board board, boolean[][] obsGrid, int activeTribeID) {
        JSONObject json = new JSONObject();
        int size = board.getSize();

        json.put("size", size);
        json.put("active_tribe_id", board.getActiveTribeID());
        json.put("actor_id_counter", 0);
        json.put("capital_ids", serializeKnownCapitalIds(board, obsGrid, activeTribeID));

        JSONArray tiles = new JSONArray();
        for (int x = 0; x < size; x++) {
            JSONArray row = new JSONArray();
            for (int y = 0; y < size; y++) {
                JSONObject tile = new JSONObject();
                boolean visible = isPositionVisible(obsGrid, x, y);
                tile.put("x", x);
                tile.put("y", y);
                tile.put("visible", visible);
                tile.put("terrain", visible ? enumName(board.getTerrainAt(x, y)) : "UNKNOWN");
                tile.put("resource", visible ? enumName(board.getResourceAt(x, y)) : null);
                tile.put("building", visible ? enumName(board.getBuildingAt(x, y)) : null);
                tile.put("city_id", visible ? board.getCityIdAt(x, y) : -1);
                int unitId = visible ? board.getUnits()[x][y] : -1;
                tile.put("unit_id", unitId);
                if (unitId != -1) {
                    tile.put("unit", serializeUnit((Unit) board.getActor(unitId)));
                }
                row.put(tile);
            }
            tiles.put(row);
        }
        json.put("tiles", tiles);

        return json;
    }

    private static JSONArray serializeKnownCapitalIds(Board board, boolean[][] obsGrid, int activeTribeID) {
        JSONArray capitalIds = new JSONArray();
        for (Tribe tribe : board.getTribes()) {
            int capitalId = tribe.getCapitalID();
            if (capitalId < 0) {
                continue;
            }
            if (tribe.getTribeId() == activeTribeID) {
                capitalIds.put(capitalId);
                continue;
            }

            Object capital = board.getActor(capitalId);
            if (capital instanceof City) {
                City city = (City) capital;
                if (isPositionVisible(obsGrid, city.getPosition().x, city.getPosition().y)) {
                    capitalIds.put(capitalId);
                }
            }
        }
        return capitalIds;
    }

    private static JSONArray serializeVisibility(boolean[][] obsGrid, int boardSize) {
        JSONArray rows = new JSONArray();
        if (obsGrid == null) {
            for (int x = 0; x < boardSize; x++) {
                JSONArray row = new JSONArray();
                for (int y = 0; y < boardSize; y++) {
                    row.put(true);
                }
                rows.put(row);
            }
            return rows;
        }

        for (int x = 0; x < obsGrid.length; x++) {
            JSONArray row = new JSONArray();
            for (int y = 0; y < obsGrid[x].length; y++) {
                row.put(obsGrid[x][y]);
            }
            rows.put(row);
        }
        return rows;
    }

    private static JSONArray serializeTribes(Board board, boolean[][] obsGrid, int activeTribeID, boolean fullObservability) {
        JSONArray tribes = new JSONArray();
        for (Tribe tribe : board.getTribes()) {
            boolean isActiveTribe = tribe.getTribeId() == activeTribeID;
            boolean includePrivateInfo = fullObservability || isActiveTribe;
            JSONObject t = new JSONObject();
            t.put("tribe_id", tribe.getTribeId());
            t.put("relation", isActiveTribe ? "SELF" : "OPPONENT");
            t.put("is_active_tribe", isActiveTribe);
            t.put("type", enumName(tribe.getType()));
            t.put("name", tribe.getName());
            t.put("stars_known", includePrivateInfo);
            t.put("stars", includePrivateInfo ? tribe.getStars() : 0);
            t.put("score", tribe.getScore());
            t.put("winner", enumName(tribe.getWinner()));
            t.put("capital_id", includePrivateInfo ? tribe.getCapitalID() : -1);
            t.put("cities", serializeCities(board, tribe, obsGrid, activeTribeID, fullObservability));
            t.put("units", serializeTribeUnits(board, tribe, obsGrid, activeTribeID, fullObservability));
            t.put("extra_units", serializeUnitIds(board, tribe.getExtraUnits(), obsGrid, activeTribeID, tribe.getTribeId()));
            t.put("connected_cities", includePrivateInfo ? toIntArray(tribe.getConnectedCities()) : new JSONArray());
            t.put("tribes_met", includePrivateInfo ? toIntArray(tribe.getTribesMet()) : new JSONArray());
            t.put("n_kills", includePrivateInfo ? tribe.getnKills() : 0);
            t.put("n_pacifist_count", includePrivateInfo ? tribe.getnPacifistCount() : 0);
            t.put("stars_sent", includePrivateInfo ? tribe.getStarsSent() : 0);
            t.put("has_declared_war", includePrivateInfo && tribe.getHasDeclaredWar());
            t.put("n_wars_declared", includePrivateInfo ? tribe.getnWarsDeclared() : 0);
            t.put("n_stars_sent", includePrivateInfo ? tribe.getnStarsSent() : 0);
            t.put("technology", includePrivateInfo ? serializeTechnologyTree(tribe.getTechTree()) : serializeHiddenTechnologyTree());
            t.put("monuments", includePrivateInfo ? serializeMonuments(tribe) : new JSONObject());
            tribes.put(t);
        }
        return tribes;
    }

    private static JSONArray serializeTribeUnits(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID, boolean fullObservability) {
        JSONArray units = new JSONArray();
        Set<Integer> seen = new HashSet<>();

        // In full-observability mode every tribe's unit list is available.
        if (fullObservability || tribe.getTribeId() == activeTribeID) {
            for (Integer cityId : tribe.getCitiesID()) {
                City city = (City) board.getActor(cityId);
                if (city == null) {
                    continue;
                }
                for (Integer unitId : city.getUnitsID()) {
                    if (unitId == null || seen.contains(unitId)) {
                        continue;
                    }
                    seen.add(unitId);
                    Unit unit = (Unit) board.getActor(unitId);
                    if (unit != null) {
                        units.put(serializeUnit(unit));
                    }
                }
            }

            for (Integer unitId : tribe.getExtraUnits()) {
                if (unitId == null || seen.contains(unitId)) {
                    continue;
                }
                seen.add(unitId);
                Unit unit = (Unit) board.getActor(unitId);
                if (unit != null) {
                    units.put(serializeUnit(unit));
                }
            }
        } else {
            // For enemy tribes, only include units in observability grid (FOW)
            for (Integer cityId : tribe.getCitiesID()) {
                City city = (City) board.getActor(cityId);
                if (city == null) {
                    continue;
                }
                for (Integer unitId : city.getUnitsID()) {
                    if (unitId == null || seen.contains(unitId)) {
                        continue;
                    }
                    seen.add(unitId);
                    Unit unit = (Unit) board.getActor(unitId);
                    if (unit != null && isPositionVisible(obsGrid, unit.getPosition().x, unit.getPosition().y)) {
                        units.put(serializeUnit(unit));
                    }
                }
            }

            for (Integer unitId : tribe.getExtraUnits()) {
                if (unitId == null || seen.contains(unitId)) {
                    continue;
                }
                seen.add(unitId);
                Unit unit = (Unit) board.getActor(unitId);
                if (unit != null && isPositionVisible(obsGrid, unit.getPosition().x, unit.getPosition().y)) {
                    units.put(serializeUnit(unit));
                }
            }
        }

        return units;
    }

    private static JSONArray serializeCities(Board board, Tribe tribe, boolean[][] obsGrid, int activeTribeID, boolean fullObservability) {
        JSONArray cities = new JSONArray();
        for (Integer cityId : tribe.getCitiesID()) {
            City city = (City) board.getActor(cityId);
            if (city == null) {
                continue;
            }
            
            // In partial-observability mode, only include enemy cities that are visible.
            if (!fullObservability && tribe.getTribeId() != activeTribeID) {
                if (!isPositionVisible(obsGrid, city.getPosition().x, city.getPosition().y)) {
                    continue;  // Skip this hidden city
                }
            }
            
            JSONObject c = new JSONObject();
            c.put("actor_id", city.getActorId());
            c.put("tribe_id", city.getTribeId());
            c.put("x", city.getPosition().x);
            c.put("y", city.getPosition().y);
            c.put("level", city.getLevel());
            c.put("population", city.getPopulation());
            c.put("population_need", city.getPopulation_need());
            c.put("is_capital", city.isCapital());
            c.put("production", city.getProduction());
            c.put("has_walls", city.hasWalls());
            c.put("bound", city.getBound());
            c.put("points_worth", city.getPointsWorth());
            c.put("unit_ids", serializeVisibleUnitIds(board, city.getUnitsID(), obsGrid, activeTribeID, tribe.getTribeId()));
            c.put("buildings", serializeBuildings(city, obsGrid, fullObservability || tribe.getTribeId() == activeTribeID));
            cities.put(c);
        }
        return cities;
    }

    private static JSONArray serializeBuildings(City city, boolean[][] obsGrid, boolean includeAll) {
        JSONArray buildings = new JSONArray();
        for (Building building : city.getBuildings()) {
            if (!includeAll && !isPositionVisible(obsGrid, building.position.x, building.position.y)) {
                continue;
            }
            JSONObject b = new JSONObject();
            b.put("x", building.position.x);
            b.put("y", building.position.y);
            b.put("type", enumName(building.type));
            b.put("city_id", building.cityId);
            if (building instanceof core.actors.Temple) {
                b.put("kind", "TEMPLE");
            } else {
                b.put("kind", "BUILDING");
            }
            buildings.put(b);
        }
        return buildings;
    }

    private static JSONArray serializeVisibleUnitIds(Board board, ArrayList<Integer> unitIds, boolean[][] obsGrid, int activeTribeID, int tribeId) {
        JSONArray ids = new JSONArray();
        Set<Integer> seen = new HashSet<>();
        for (Integer unitId : unitIds) {
            if (unitId == null || seen.contains(unitId)) {
                continue;
            }
            seen.add(unitId);
            Unit unit = (Unit) board.getActor(unitId);
            if (unit == null) {
                continue;
            }
            if (tribeId != activeTribeID && !isPositionVisible(obsGrid, unit.getPosition().x, unit.getPosition().y)) {
                continue;
            }
            ids.put(unitId);
        }
        return ids;
    }

    private static JSONObject serializeUnit(Unit unit) {
        if (unit == null) {
            return null;
        }

        JSONObject json = new JSONObject();
        json.put("actor_id", unit.getActorId());
        json.put("tribe_id", unit.getTribeId());
        json.put("city_id", unit.getCityId());
        json.put("x", unit.getPosition().x);
        json.put("y", unit.getPosition().y);
        json.put("type", enumName(unit.getType()));
        json.put("status", enumName(unit.getStatus()));
        json.put("current_hp", unit.getCurrentHP());
        json.put("max_hp", unit.getMaxHP());
        json.put("kills", unit.getKills());
        json.put("is_veteran", unit.isVeteran());
        json.put("atk", unit.ATK);
        json.put("def", unit.DEF);
        json.put("mov", unit.MOV);
        json.put("range", unit.RANGE);
        json.put("cost", unit.COST);
        json.put("has_moved", unit.getStatus() == Types.TURN_STATUS.MOVED || unit.getStatus() == Types.TURN_STATUS.MOVED_AND_ATTACKED);
        json.put("has_attacked", unit.getStatus() == Types.TURN_STATUS.ATTACKED || unit.getStatus() == Types.TURN_STATUS.MOVED_AND_ATTACKED);
        return json;
    }

    private static JSONArray serializeUnitIds(Board board, ArrayList<Integer> unitIds, boolean[][] obsGrid, int activeTribeID, int tribeId) {
        JSONArray units = new JSONArray();
        Set<Integer> seen = new HashSet<>();
        for (Integer unitId : unitIds) {
            if (unitId == null || seen.contains(unitId)) {
                continue;
            }
            seen.add(unitId);
            Unit unit = (Unit) board.getActor(unitId);
            if (unit != null) {
                // Check visibility for enemy units
                if (tribeId != activeTribeID) {
                    if (!isPositionVisible(obsGrid, unit.getPosition().x, unit.getPosition().y)) {
                        continue;
                    }
                }
                units.put(serializeUnit(unit));
            }
        }
        return units;
    }

    private static boolean isPositionVisible(boolean[][] obsGrid, int x, int y) {
        /**
         * Check if a position is visible in the observability grid.
         * Returns true if:
         * 1. obsGrid is null (no FOW restriction)
         * 2. Position is within grid bounds and marked as observable
         */
        if (obsGrid == null) {
            return true;
        }
        if (x < 0 || y < 0 || x >= obsGrid.length) {
            return false;
        }
        if (y >= obsGrid[0].length) {
            return false;
        }
        return obsGrid[x][y];
    }

    public static JSONObject encodeActionComponents(Action action, GameState gs) {
        JSONObject components = new JSONObject();
        int actionTypeIndex = actionTypeIndexFromSchema(action.getActionType());
        components.put("action_type_index", actionTypeIndex);
        components.put("source_actor_index", sourceActorIndex(action));
        components.put("target_actor_index", targetActorIndex(action, gs));
        components.put("param_index", paramIndex(action));
        return components;
    }

    private static JSONArray serializeActions(ArrayList<Action> actions) {
        return serializeActions(actions, null);
    }

    private static JSONArray serializeActions(ArrayList<Action> actions, GameState gs) {
        JSONArray serialized = new JSONArray();
        for (int i = 0; i < actions.size(); i++) {
            Action action = actions.get(i);
            JSONObject json = new JSONObject();
            json.put("index", i);
            json.put("action_type", enumName(action.getActionType()));
            json.put("action_type_index", actionTypeIndexFromSchema(action.getActionType()));
            json.put("encoded_components", encodeActionComponents(action, gs));
            json.put("class_name", action.getClass().getSimpleName());
            json.put("description", String.valueOf(action));
            serialized.put(json);
        }
        return serialized;
    }

    private static int sourceActorIndex(Action action) {
        if (action instanceof core.actions.unitactions.UnitAction) {
            return ((core.actions.unitactions.UnitAction) action).getUnitId();
        }
        if (action instanceof core.actions.cityactions.CityAction) {
            return ((core.actions.cityactions.CityAction) action).getCityId() + 100;
        }
        return 0;
    }

    private static int targetActorIndex(Action action, GameState gs) {
        if (action instanceof core.actions.unitactions.Move) {
            utils.Vector2d destination = ((core.actions.unitactions.Move) action).getDestination();
            return destination == null ? 0 : positionToTargetIndex(destination.x, destination.y);
        }
        if (action instanceof core.actions.unitactions.Attack) {
            return ((core.actions.unitactions.Attack) action).getTargetId() + 121;
        }
        if (action instanceof core.actions.unitactions.Convert) {
            return ((core.actions.unitactions.Convert) action).getTargetId() + 121;
        }
        if (action instanceof core.actions.unitactions.Capture) {
            core.actions.unitactions.Capture capture = (core.actions.unitactions.Capture) action;
            if (capture.getCaptureType() == core.Types.TERRAIN.CITY) {
                return capture.getTargetCity() + 221;
            }
            if (capture.getCaptureType() == core.Types.TERRAIN.VILLAGE) {
                if (gs != null) {
                    core.actors.units.Unit unit = (core.actors.units.Unit) gs.getActor(capture.getUnitId());
                    if (unit != null && unit.getPosition() != null) {
                        return positionToTargetIndex(unit.getPosition().x, unit.getPosition().y);
                    }
                }
            }
            return 0;
        }
        if (action instanceof core.actions.tribeactions.BuildRoad) {
            utils.Vector2d position = ((core.actions.tribeactions.BuildRoad) action).getPosition();
            return position == null ? 0 : positionToTargetIndex(position.x, position.y);
        }
        if (action instanceof core.actions.tribeactions.SendStars) {
            return ((core.actions.tribeactions.SendStars) action).getTargetID() + 271;
        }
        if (action instanceof core.actions.tribeactions.DeclareWar) {
            return ((core.actions.tribeactions.DeclareWar) action).getTargetID() + 271;
        }
        if (action instanceof core.actions.cityactions.Build) {
            utils.Vector2d targetPos = ((core.actions.cityactions.Build) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.BurnForest) {
            utils.Vector2d targetPos = ((core.actions.cityactions.BurnForest) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.ClearForest) {
            utils.Vector2d targetPos = ((core.actions.cityactions.ClearForest) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.Destroy) {
            utils.Vector2d targetPos = ((core.actions.cityactions.Destroy) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.GrowForest) {
            utils.Vector2d targetPos = ((core.actions.cityactions.GrowForest) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.LevelUp) {
            utils.Vector2d targetPos = ((core.actions.cityactions.LevelUp) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.cityactions.ResourceGathering) {
            utils.Vector2d targetPos = ((core.actions.cityactions.ResourceGathering) action).getTargetPos();
            return targetPos == null ? 0 : positionToTargetIndex(targetPos.x, targetPos.y);
        }
        if (action instanceof core.actions.unitactions.Examine) {
            core.actions.unitactions.UnitAction unitAction = (core.actions.unitactions.UnitAction) action;
            if (gs != null) {
                core.actors.units.Unit unit = (core.actors.units.Unit) gs.getActor(unitAction.getUnitId());
                if (unit != null && unit.getPosition() != null) {
                    return positionToTargetIndex(unit.getPosition().x, unit.getPosition().y);
                }
            }
            return 0;
        }
        return 0;
    }

    private static int paramIndex(Action action) {
        if (action instanceof core.actions.tribeactions.SendStars) {
            return ((core.actions.tribeactions.SendStars) action).getNumStars();
        }
        if (action instanceof core.actions.tribeactions.ResearchTech) {
            Types.TECHNOLOGY tech = ((core.actions.tribeactions.ResearchTech) action).getTech();
            return tech == null ? 0 : tech.ordinal();
        }
        if (action instanceof core.actions.cityactions.Build) {
            Types.BUILDING buildingType = ((core.actions.cityactions.Build) action).getBuildingType();
            return buildingType == null ? 0 : buildingType.ordinal();
        }
        if (action instanceof core.actions.cityactions.Spawn) {
            Types.UNIT unitType = ((core.actions.cityactions.Spawn) action).getUnitType();
            return unitType == null ? 0 : unitType.ordinal();
        }
        if (action instanceof core.actions.cityactions.LevelUp) {
            Types.CITY_LEVEL_UP bonus = ((core.actions.cityactions.LevelUp) action).getBonus();
            return bonus == null ? 0 : bonus.ordinal();
        }
        if (action instanceof core.actions.cityactions.ResourceGathering) {
            Types.RESOURCE resource = ((core.actions.cityactions.ResourceGathering) action).getResource();
            return resource == null ? 0 : resource.ordinal();
        }
        if (action instanceof core.actions.unitactions.Examine) {
            Types.EXAMINE_BONUS bonus = ((core.actions.unitactions.Examine) action).getBonus();
            return bonus == null ? 0 : bonus.ordinal();
        }
        return 0;
    }

    private static int positionToTargetIndex(int x, int y) {
        int boardSize = 11;
        JSONObject schema = loadSchema();
        if (schema != null) {
            boardSize = schema.optInt("board_size", boardSize);
        }
        return x * boardSize + y + 1;
    }

    private static JSONObject loadSchema() {
        if (cachedSchema != null) {
            return cachedSchema;
        }

        java.io.File schemaFile = new java.io.File("../az_api/action_space_schema.json");
        if (!schemaFile.exists()) {
            return null;
        }

        try (java.io.FileReader reader = new java.io.FileReader(schemaFile)) {
            StringBuilder sb = new StringBuilder();
            char[] buffer = new char[4096];
            int read;
            while ((read = reader.read(buffer)) != -1) {
                sb.append(buffer, 0, read);
            }
            cachedSchema = new JSONObject(sb.toString());
            return cachedSchema;
        } catch (IOException e) {
            System.out.println("[PythonBridge] error loading schema: " + e.getMessage());
            return null;
        }
    }

    private static Integer actionTypeIndexFromSchema(Types.ACTION actionType) {
        if (actionType == null) {
            return null;
        }

        JSONObject schema = loadSchema();
        if (schema == null) {
            return null;
        }

        try {
            JSONObject components = schema.getJSONObject("components");
            JSONObject actionTypeComponent = components.getJSONObject("action_type");
            JSONObject indexMap = actionTypeComponent.getJSONObject("index_map");
            String key = actionType.name();
            if (indexMap.has(key)) {
                return indexMap.getInt(key);
            }
        } catch (Exception e) {
            System.out.println("[PythonBridge] error reading action type index: " + e.getMessage());
        }

        return null;
    }

    private static JSONObject serializeTechnologyTree(TechnologyTree techTree) {
        JSONObject json = new JSONObject();
        boolean[] researched = techTree.getResearched();
        JSONArray researchedTechs = new JSONArray();
        JSONArray researchedFlags = new JSONArray();

        for (Types.TECHNOLOGY technology : Types.TECHNOLOGY.values()) {
            boolean isResearched = researched[technology.ordinal()];
            researchedFlags.put(isResearched);
            if (isResearched) {
                researchedTechs.put(technology.name());
            }
        }

        json.put("researched", researchedTechs);
        json.put("researched_flags", researchedFlags);
        json.put("everything_researched", techTree.isEverythingResearched());
        json.put("num_researched", techTree.getNumResearched());
        return json;
    }

    private static JSONObject serializeHiddenTechnologyTree() {
        JSONObject json = new JSONObject();
        JSONArray researchedTechs = new JSONArray();
        JSONArray researchedFlags = new JSONArray();

        for (Types.TECHNOLOGY ignored : Types.TECHNOLOGY.values()) {
            researchedFlags.put(false);
        }

        json.put("researched", researchedTechs);
        json.put("researched_flags", researchedFlags);
        json.put("everything_researched", false);
        json.put("num_researched", 0);
        json.put("hidden", true);
        return json;
    }

    private static JSONObject serializeMonuments(Tribe tribe) {
        JSONObject json = new JSONObject();
        for (Types.BUILDING building : tribe.getMonuments().keySet()) {
            json.put(building.name(), tribe.getMonuments().get(building).name());
        }
        return json;
    }

    private static String enumName(Enum<?> value) {
        return value == null ? null : value.name();
    }

    private static JSONArray toIntArray(int[] values) {
        JSONArray array = new JSONArray();
        if (values == null) {
            return array;
        }
        for (int value : values) {
            array.put(value);
        }
        return array;
    }

    private static JSONArray toIntArray(ArrayList<Integer> values) {
        JSONArray array = new JSONArray();
        if (values == null) {
            return array;
        }
        for (Integer value : values) {
            array.put(value);
        }
        return array;
    }

}
