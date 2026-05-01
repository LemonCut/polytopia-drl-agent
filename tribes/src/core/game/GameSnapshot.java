package core.game;

import core.Types;
import core.actors.Actor;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Battleship;
import core.actors.units.Boat;
import core.actors.units.Ship;
import core.actors.units.Unit;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

final class GameSnapshot {

    private GameSnapshot() {
    }

    static JSONObject build(GameState gs, long seed) {
        JSONObject game = new JSONObject();
        Board gameBoard = gs.getBoard();

        JSONObject board = new JSONObject();
        JSONArray terrain2D = new JSONArray();
        JSONArray resource2D = new JSONArray();
        JSONArray unit2D = new JSONArray();
        JSONArray city2D = new JSONArray();
        JSONArray building2D = new JSONArray();
        JSONArray network2D = new JSONArray();

        for (int i = 0; i < gameBoard.getSize(); i++) {
            JSONArray terrain = new JSONArray();
            JSONArray resource = new JSONArray();
            JSONArray units = new JSONArray();
            JSONArray cities = new JSONArray();
            JSONArray networks = new JSONArray();
            JSONArray buildings = new JSONArray();

            for (int j = 0; j < gameBoard.getSize(); j++) {
                terrain.put(gameBoard.getTerrainAt(i, j).getKey());
                resource.put(gameBoard.getResourceAt(i, j) != null ? gameBoard.getResourceAt(i, j).getKey() : -1);
                units.put(gameBoard.getUnitIDAt(i, j));
                cities.put(gameBoard.getCityIdAt(i, j));
                buildings.put(gameBoard.getBuildingAt(i, j) != null ? gameBoard.getBuildingAt(i, j).getKey() : -1);
                networks.put(gameBoard.getNetworkTilesAt(i, j));
            }

            terrain2D.put(terrain);
            resource2D.put(resource);
            unit2D.put(units);
            city2D.put(cities);
            building2D.put(buildings);
            network2D.put(networks);
        }

        board.put("terrain", terrain2D);
        board.put("resource", resource2D);
        board.put("unitID", unit2D);
        board.put("cityID", city2D);
        board.put("building", building2D);
        board.put("network", network2D);
        board.put("actorIDcounter", gameBoard.getActorIDcounter());
        game.put("board", board);

        JSONObject unit = new JSONObject();
        for (Unit u : getAllUnits(gameBoard)) {
            JSONObject uInfo = new JSONObject();
            uInfo.put("type", u.getType().getKey());
            if (u.getType() == Types.UNIT.BOAT) {
                uInfo.put("baseLandType", ((Boat) u).getBaseLandUnit().getKey());
            } else if (u.getType() == Types.UNIT.SHIP) {
                uInfo.put("baseLandType", ((Ship) u).getBaseLandUnit().getKey());
            } else if (u.getType() == Types.UNIT.BATTLESHIP) {
                uInfo.put("baseLandType", ((Battleship) u).getBaseLandUnit().getKey());
            }
            uInfo.put("x", u.getPosition().x);
            uInfo.put("y", u.getPosition().y);
            uInfo.put("kill", u.getKills());
            uInfo.put("isVeteran", u.isVeteran());
            uInfo.put("cityID", u.getCityId());
            uInfo.put("tribeId", u.getTribeId());
            uInfo.put("currentHP", u.getCurrentHP());
            unit.put(String.valueOf(u.getActorId()), uInfo);
        }
        game.put("unit", unit);

        JSONObject city = new JSONObject();
        for (City c : getAllCities(gameBoard)) {
            JSONObject cInfo = new JSONObject();
            cInfo.put("x", c.getPosition().x);
            cInfo.put("y", c.getPosition().y);
            cInfo.put("tribeID", c.getTribeId());
            cInfo.put("population_need", c.getPopulation_need());
            cInfo.put("bound", c.getBound());
            cInfo.put("level", c.getLevel());
            cInfo.put("isCapital", c.isCapital());
            cInfo.put("population", c.getPopulation());
            cInfo.put("production", c.getProduction());
            cInfo.put("hasWalls", c.hasWalls());
            cInfo.put("pointsWorth", c.getPointsWorth());

            JSONArray buildingList = new JSONArray();
            if (c.getBuildings() != null) {
                for (core.actors.Building b : c.getBuildings()) {
                    JSONObject bInfo = new JSONObject();
                    bInfo.put("x", b.position.x);
                    bInfo.put("y", b.position.y);
                    bInfo.put("type", b.type.getKey());
                    if (b.type.isTemple()) {
                        core.actors.Temple t = (core.actors.Temple) b;
                        bInfo.put("level", t.getLevel());
                        bInfo.put("turnsToScore", t.getTurnsToScore());
                    }
                    buildingList.put(bInfo);
                }
            }
            cInfo.put("buildings", buildingList);
            cInfo.put("units", c.getUnitsID());
            city.put(String.valueOf(c.getActorId()), cInfo);
        }
        game.put("city", city);

        JSONObject tribesINFO = new JSONObject();
        for (Tribe t : gameBoard.getTribes()) {
            JSONObject tribeInfo = new JSONObject();
            tribeInfo.put("citiesID", t.getCitiesID());
            tribeInfo.put("capitalID", t.getCapitalID());
            tribeInfo.put("type", t.getType().getKey());

            JSONObject techINFO = new JSONObject();
            techINFO.put("researched", t.getTechTree().getResearched());
            techINFO.put("everythingResearched", t.getTechTree().isEverythingResearched());
            tribeInfo.put("technology", techINFO);
            tribeInfo.put("star", t.getStars());
            tribeInfo.put("winner", t.getWinner().getKey());
            tribeInfo.put("score", t.getScore());
            tribeInfo.put("obsGrid", t.getObsGrid());
            tribeInfo.put("connectedCities", t.getConnectedCities());

            HashMap<Types.BUILDING, Types.BUILDING.MONUMENT_STATUS> monuments = t.getMonuments();
            JSONObject monumentInfo = new JSONObject();
            for (Types.BUILDING key : monuments.keySet()) {
                monumentInfo.put(String.valueOf(key.getKey()), monuments.get(key).getKey());
            }
            tribeInfo.put("monuments", monumentInfo);
            JSONArray tribesMetInfo = new JSONArray();
            for (Integer tribeId : t.getTribesMet()) {
                tribesMetInfo.put(tribeId);
            }
            tribeInfo.put("tribesMet", tribesMetInfo);
            tribeInfo.put("extraUnits", t.getExtraUnits());
            tribeInfo.put("nKills", t.getnKills());
            tribeInfo.put("nPacifistCount", t.getnPacifistCount());
            tribesINFO.put(String.valueOf(t.getActorId()), tribeInfo);
        }
        game.put("tribes", tribesINFO);

        game.put("seed", seed);
        game.put("tick", gs.getTick());
        game.put("gameIsOver", gs.isGameOver());
        game.put("activeTribeID", gs.getActiveTribeID());
        game.put("gameMode", gs.getGameMode().getKey());
        return game;
    }

    private static ArrayList<City> getAllCities(Board board) {
        Tribe[] tribes = board.getTribes();
        ArrayList<City> cityActors = new ArrayList<>();
        for (Tribe t : tribes) {
            for (Integer cityId : t.getCitiesID()) {
                cityActors.add((City) board.getActor(cityId));
            }
        }
        return cityActors;
    }

    private static ArrayList<Unit> getAllUnits(Board board) {
        Tribe[] tribes = board.getTribes();
        ArrayList<Unit> unitActors = new ArrayList<>();
        for (Tribe t : tribes) {
            for (Integer cityId : t.getCitiesID()) {
                City c = (City) board.getActor(cityId);
                for (Integer unitId : c.getUnitsID()) {
                    unitActors.add((Unit) board.getActor(unitId));
                }
            }

            for (Integer unitId : t.getExtraUnits()) {
                unitActors.add((Unit) board.getActor(unitId));
            }
        }
        return unitActors;
    }
}