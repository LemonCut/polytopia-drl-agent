package players.azmcts;

import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.game.GameState;
import core.Types;
import org.json.JSONObject;
import players.PythonBridge;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;
import static core.Types.ACTION.*;

class SingleTreeNode
{
    private MCTSParams params;

    private SingleTreeNode root;
    private SingleTreeNode parent;
    private SingleTreeNode[] children;
    private double totValue;
    private int nVisits;
    private Random m_rnd;
    private int m_depth;
    private double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private int fmCallsCount;
    private int playerID;

    private ArrayList<Action> actions;
    private GameState state;

    // Neural-guided MCTS cache (PUCT priors + value)
    private boolean nnEvaluated;
    private double[] nnPriors;
    private double nnValue;

    private GameState rootState;
    private StateHeuristic rootStateHeuristic;

    private boolean rootNoiseApplied = false;
    private double[] rootNoisyPriors;

    //From MCTSPlayer
    SingleTreeNode(MCTSParams p, Random rnd, int num_actions, ArrayList<Action> actions, int playerID) {
        this(p, null, rnd, num_actions, actions, null, playerID, null, null);
    }

    private SingleTreeNode(MCTSParams p, SingleTreeNode parent, Random rnd, int num_actions,
                           ArrayList<Action> actions, StateHeuristic sh, int playerID, SingleTreeNode root, GameState state) {
        this.params = p;
        this.fmCallsCount = 0;
        this.parent = parent;
        this.m_rnd = rnd;
        this.actions = actions;
        this.root = root;
        children = new SingleTreeNode[num_actions];
        totValue = 0.0;
        this.playerID = playerID;
        this.state = state;
        this.nnEvaluated = false;
        this.nnPriors = null;
        this.nnValue = 0.0;
        if(parent != null) {
            m_depth = parent.m_depth + 1;
            this.rootStateHeuristic = sh;
        }
        else {
            m_depth = 0;
        }

    }

    void setRootGameState(SingleTreeNode root, GameState gs, ArrayList<Integer> allIDs)
    {
        this.state = gs;
        this.root = root;
        this.rootState = gs;
        this.rootStateHeuristic = params.getStateHeuristic(playerID, allIDs);
    }


    void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int numIters = 0;

        int remainingLimit = 5;
        boolean stop = false;

        while(!stop){
//            System.out.println("------- " + root.actions.size() + " -------");
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            SingleTreeNode selected = treePolicy();
            double delta = selected.rollOut();
            backUp(selected, delta);
            numIters++;

            //Stopping condition
            if(params.stop_type == params.STOP_TIME) {
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
                avgTimeTaken  = acumTimeTaken/numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            }else if(params.stop_type == params.STOP_ITERATIONS) {
                stop = numIters >= params.num_iterations;
            }else if(params.stop_type == params.STOP_FMCALLS)
            {
                stop = fmCallsCount > params.num_fmcalls;
            }
        }
    }

    private SingleTreeNode treePolicy() {

        SingleTreeNode cur = this;

        while (!cur.state.isGameOver() /*&& state.getAllAvailableActions().size() > 1 */ && cur.m_depth < params.ROLLOUT_LENGTH)
        {
            ArrayList<Action> availableActions = cur.getAvailableActionsForNode();
            if (availableActions == null || availableActions.isEmpty()) {
                return cur;
            }

            int forcedEndTurn = cur.forcedEndTurnAction(availableActions);
            if (forcedEndTurn != -1) {
                if (cur.children[forcedEndTurn] == null) {
                    return cur.expandAction(forcedEndTurn, availableActions);
                }
                cur = cur.children[forcedEndTurn];
                continue;
            }

            if (params.NEURAL_PRIORS) {
                cur.ensureNeuralEvaluated(availableActions);
                int actionIdx = cur.selectPuctAction(availableActions);
                if (actionIdx < 0) {
                    return cur;
                }
                if (cur.children[actionIdx] == null) {
                    return cur.expandAction(actionIdx, availableActions);
                }
                cur = cur.children[actionIdx];
            } else if (cur.notFullyExpanded()) {
                return cur.expand();
            } else {
                cur = cur.uct();
            }
        }

        return cur;
    }

    private int forcedEndTurnAction(ArrayList<Action> availableActions)
    {
        if (!params.FORCE_END_TURN_IN_SEARCH || params.FORCE_TURN_END <= 0) {
            return -1;
        }
        EndTurn endTurn = new EndTurn(state.getActiveTribeID());
        int depth = this.m_depth;
        boolean willForceEnd = (depth > 0 && (depth % params.FORCE_TURN_END) == 0) && endTurn.isFeasible(state);
        if(!willForceEnd)
            return -1; //Not the time, or not available.
        int actionIdx = 0;
        while(actionIdx < availableActions.size())
        {
            Action act = availableActions.get(actionIdx);
            if(act.getActionType() == END_TURN)
            {
                //Here's the end turn, return it's index.
                return actionIdx;
            }else actionIdx++;
        }

        //This should not happen, but EndTurn is not available here.
        return -1;
    }

    private SingleTreeNode expandAction(int actionIdx, ArrayList<Action> availableActions) {
        GameState nextState = state.copy();
        ArrayList<Action> nextActions = advance(nextState, availableActions.get(actionIdx), true);
        SingleTreeNode tn = new SingleTreeNode(params, this, this.m_rnd, nextActions.size(),
                nextActions, rootStateHeuristic, this.playerID, this.m_depth == 0 ? this : this.root, nextState);
        children[actionIdx] = tn;
        return tn;
    }

    private SingleTreeNode expand() {

        ArrayList<Action> availableActions = getAvailableActionsForNode();
        int bestAction = forcedEndTurnAction(availableActions);
        if(bestAction == -1) {
            bestAction = selectUnexpandedAction(availableActions);
        }

        return expandAction(bestAction, availableActions);
    }



    private ArrayList<Action> advance(GameState gs, Action act, boolean computeActions)
    {
        gs.advance(act, computeActions);
        root.fmCallsCount++;
        return availableActionsForState(gs);
    }

    private int selectPuctAction(ArrayList<Action> availableActions) {
        boolean rootPlayerToMove = (state.getActiveTribeID() == this.playerID);
        double bestValue = rootPlayerToMove ? -Double.MAX_VALUE : Double.MAX_VALUE;
        int selected = -1;
        double parentVisits = Math.max(1.0, this.nVisits);

        for (int i = 0; i < this.children.length; ++i) {
            SingleTreeNode child = children[i];
            int childVisits = child == null ? 0 : child.nVisits;
            double q = (child == null || childVisits == 0)
                    ? 0.0
                    : child.totValue / (childVisits + params.epsilon);
            double prior = priorForAction(i, availableActions.size());
            double u = params.CPUCT * prior * Math.sqrt(parentVisits) / (1.0 + childVisits);
            double score = rootPlayerToMove ? (q + u) : (q - u);
            score = noise(score, params.epsilon, this.m_rnd.nextDouble());

            if ((rootPlayerToMove && score > bestValue) || (!rootPlayerToMove && score < bestValue)) {
                selected = i;
                bestValue = score;
            }
        }

        return selected;
    }

    private double priorForAction(int actionIdx, int numActions) {
        if (nnPriors != null && actionIdx >= 0 && actionIdx < nnPriors.length) {
            double prior = nnPriors[actionIdx];
            if (Double.isFinite(prior) && prior > 0.0) {
                return prior;
            }
        }
        if (numActions <= 0) {
            return 0.0;
        }
        return 1.0 / numActions;
    }

    private int selectUnexpandedAction(ArrayList<Action> availableActions) {
        if (params.NEURAL_PRIORS) {
            ensureNeuralEvaluated(availableActions);
        }

        double bestScore = -Double.MAX_VALUE;
        int picked = -1;
        for (int i = 0; i < children.length; i++) {
            if (children[i] != null) {
                continue;
            }
            double score = priorForAction(i, availableActions.size());
            score = noise(score, params.epsilon, this.m_rnd.nextDouble());
            if (score > bestScore) {
                bestScore = score;
                picked = i;
            }
        }
        if (picked == -1) {
            picked = 0;
        }
        return picked;
    }


    private SingleTreeNode uct() {

        SingleTreeNode selected;
        boolean IamMoving = (state.getActiveTribeID() == this.playerID);
        ArrayList<Action> availableActions = getAvailableActionsForNode();
        int bestAction = forcedEndTurnAction(availableActions);
        if(bestAction == -1)
        {
            //No end turn, use uct.
            if (params.NEURAL_PRIORS) {
                ensureNeuralEvaluated(availableActions);
            }

            int which = -1;
            double bestValue = IamMoving ? -Double.MAX_VALUE : Double.MAX_VALUE;

            for(int i = 0; i < this.children.length; ++i)
            {
                SingleTreeNode child = children[i];

                double q = child.totValue / (child.nVisits + params.epsilon);
                double score;
                if (params.NEURAL_PRIORS && nnPriors != null && i < nnPriors.length) {
                    double u = params.CPUCT * nnPriors[i] * Math.sqrt(this.nVisits + 1.0) / (1.0 + child.nVisits);
                    score = IamMoving ? (q + u) : (q - u);
                } else {
                    double childValue = normalise(q, bounds[0], bounds[1]);
                    double uctValue = childValue +
                            params.K * Math.sqrt(Math.log(this.nVisits + 1) / (child.nVisits + params.epsilon));
                    score = uctValue;
                }

                score = noise(score, params.epsilon, this.m_rnd.nextDouble());

                if ((IamMoving && score > bestValue) || (!IamMoving && score < bestValue)){
                    which = i;
                    bestValue = score;
                }
            }

            if (which == -1)
            {
                //if(this.children.length == 0)
                System.out.println("Warning! couldn't find the best UCT value " + which + " : " + this.children.length + " " +
                //throw new RuntimeException("Warning! couldn't find the best UCT value " + which + " : " + this.children.length + " " +
                        + bounds[0] + " " + bounds[1]);
                System.out.print(this.m_depth + ", AmIMoving? " + IamMoving + ";");
                System.out.println("; selected: " + which);

                which = m_rnd.nextInt(children.length);
            }

            selected = children[which];

//            System.out.print(this.m_depth + ", AmIMoving? " + IamMoving + ";");
//            for(int i = 0; i < this.children.length; ++i)
//                System.out.printf(" %f2", vals[i]);
//            System.out.println("; selected: " + which);

        }else
        {
            selected = children[bestAction];
        }

        //Roll the state. This is closed loop, we don't advance the state. We can't do open loop here because the
        // number of actions available on a state depend on the state itself, and random events triggered by multiple
        // runs over the same tree node would have different outcomes (i.e Examine ruins).
        //advance(state, actions.get(selected.childIdx), true);

        return selected;
    }

    private double rollOut()
    {
        if (params.NEURAL_VALUE) {
            if (state.isGameOver()) {
                try {
                    Types.RESULT winner = state.getTribe(playerID).getWinner();
                    if (winner == Types.RESULT.WIN) return 1.0;
                    if (winner == Types.RESULT.LOSS) return -1.0;
                } catch (Exception ignored) {
                    // fall through
                }
                return 0.0;
            }

            try {
                ArrayList<Action> availableActions = getAvailableActionsForNode();
                ensureNeuralEvaluated(availableActions);
                return nnValue;
            } catch (Exception e) {
                // If Python bridge is down, fall back to heuristic evaluation.
            }
        }

        if(params.ROLOUTS_ENABLED) {
            GameState rolloutState = state.copy();
            int thisDepth = this.m_depth;
            while (!finishRollout(rolloutState, thisDepth)) {
                ArrayList<Action> rolloutActions = availableActionsForState(rolloutState);
                int bestAction = forcedEndTurnActionForRollout(rolloutState, rolloutActions, thisDepth);
                Action next = (bestAction != -1) ? rolloutActions.get(bestAction) : rolloutActions.get(m_rnd.nextInt(rolloutActions.size()));
                advance(rolloutState, next, true);
                thisDepth++;
            }
            return normalise(this.rootStateHeuristic.evaluateState(root.rootState, rolloutState), 0, 1);
        }

        return normalise(this.rootStateHeuristic.evaluateState(root.rootState, this.state), 0, 1);
    }

    private ArrayList<Action> getAvailableActionsForNode() {
        if (actions != null) {
            return actions;
        }
        return availableActionsForState(state);
    }

    private ArrayList<Action> availableActionsForState(GameState gs) {
        ArrayList<Action> availableActions = gs.getAllAvailableActions();
        if (!params.MASK_SEND_STARS || availableActions == null || availableActions.isEmpty()) {
            return availableActions;
        }

        ArrayList<Action> filtered = new ArrayList<>();
        for (Action action : availableActions) {
            if (action.getActionType() != SEND_STARS) {
                filtered.add(action);
            }
        }

        return filtered.isEmpty() ? availableActions : filtered;
    }

    private int forcedEndTurnActionForRollout(GameState rolloutState, ArrayList<Action> availableActions, int depth) {
        if (!params.FORCE_END_TURN_IN_SEARCH || params.FORCE_TURN_END <= 0) {
            return -1;
        }
        EndTurn endTurn = new EndTurn(rolloutState.getActiveTribeID());
        boolean willForceEnd = (depth > 0 && (depth % params.FORCE_TURN_END) == 0) && endTurn.isFeasible(rolloutState);
        if (!willForceEnd) {
            return -1;
        }

        for (int actionIdx = 0; actionIdx < availableActions.size(); actionIdx++) {
            Action act = availableActions.get(actionIdx);
            if (act.getActionType() == END_TURN) {
                return actionIdx;
            }
        }
        return -1;
    }

    private void ensureNeuralEvaluated(ArrayList<Action> availableActions) {
        if (nnEvaluated) {
            return;
        }

        if ((!params.NEURAL_PRIORS) && (!params.NEURAL_VALUE)) {
            nnEvaluated = true;
            return;
        }

        try {
            JSONObject resp = PythonBridge.queryPolicyJson(state, availableActions, PythonBridge.policyUrl(playerID));
            if (!"success".equals(resp.optString("status", "error"))) {
                System.out.println("[MCTS] NN query failed: " + resp.optString("error", "unknown"));
                nnPriors = null;
                nnValue = 0.0;
                nnEvaluated = true;
                return;
            }

            if (params.NEURAL_PRIORS) {
                nnPriors = PythonBridge.actionPriorsFromPolicy(availableActions, state, resp);

                if (params.DIRICHLET_ROOT_NOISE && m_depth == 0 && !rootNoiseApplied) {
                    nnPriors = applyRootDirichlet(nnPriors);
                    rootNoiseApplied = true;
                }
            }

            if (params.NEURAL_VALUE) {
                double v = resp.optDouble("value", 0.0);
                // /query returns value from the perspective of the active player.
                // Convert to root player's perspective.
                int active = state.getActiveTribeID();
                nnValue = (active == playerID) ? v : -v;
            }

            nnEvaluated = true;
        } catch (Exception e) {
            System.out.println("[MCTS] NN query exception: " + e.getMessage());
            nnPriors = null;
            nnValue = 0.0;
            nnEvaluated = true;
        }
    }
    private boolean finishRollout(GameState rollerState, int depth)
    {
        if (depth >= params.ROLLOUT_LENGTH)      //rollout end condition.
            return true;

        //end of game
        return rollerState.isGameOver();
    }


    private void backUp(SingleTreeNode node, double result)
    {
        SingleTreeNode n = node;
        while(n != null)
        {
            n.nVisits++;
            n.totValue += result;
            if (result < n.bounds[0]) {
                n.bounds[0] = result;
            }
            if (result > n.bounds[1]) {
                n.bounds[1] = result;
            }
            n = n.parent;
        }
    }


    int mostVisitedAction() {
        int selected = -1;
        int bestVisits = -1;
        int ties = 0;

        for (int i = 0; i < children.length; i++) {
            if (children[i] == null) {
                continue;
            }

            int childVisits = children[i].nVisits;
            if (childVisits > bestVisits) {
                bestVisits = childVisits;
                selected = i;
                ties = 1;
            } else if (childVisits == bestVisits) {
                ties++;
                if (m_rnd.nextInt(ties) == 0) {
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            selected = 0;
        }

        return selected;
    }

    int sampleVisitCountAction() {
        int totalVisits = 0;
        for (SingleTreeNode child : children) {
            if (child != null && child.nVisits > 0) {
                totalVisits += child.nVisits;
            }
        }

        if (totalVisits <= 0) {
            return mostVisitedAction();
        }

        int threshold = m_rnd.nextInt(totalVisits);
        int cumulative = 0;
        for (int i = 0; i < children.length; i++) {
            SingleTreeNode child = children[i];
            if (child == null || child.nVisits <= 0) {
                continue;
            }
            cumulative += child.nVisits;
            if (threshold < cumulative) {
                return i;
            }
        }

        return mostVisitedAction();
    }

    private int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        for (int i=0; i<children.length; i++) {

            if(children[i] != null) {
                double childValue = children[i].totValue / (children[i].nVisits + params.epsilon);
                childValue = noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    private boolean notFullyExpanded() {
        for (SingleTreeNode tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }

    private double normalise(double a_value, double a_min, double a_max)
    {
        if(a_min < a_max)
            return (a_value - a_min)/(a_max - a_min);
        else    // if bounds are invalid, then return same value
            return a_value;
    }

    private double noise(double input, double epsilon, double random)
    {
        return (input + epsilon) * (1.0 + epsilon * (random - 0.5));
    }

    int[] getVisitCounts() {
        int[] visits = new int[children.length];
        for (int i = 0; i < children.length; i++) {
            if (children[i] != null) {
                visits[i] = children[i].nVisits;
            }
        }
        return visits;
    }

    double getRootValue() {
        if (nVisits <= 0) {
            return 0.0;
        }
        return totValue / (nVisits + params.epsilon);
    }

    private double[] dirichletNoise(int size, double alpha) {
        double[] noise = new double[size];
        double sum = 0.0;

        for (int i = 0; i < size; i++) {
            noise[i] = sampleGamma(alpha, 1.0);
            sum += noise[i];
        }

        for (int i = 0; i < size; i++) {
            noise[i] /= sum;
        }

        return noise;
    }
    private double sampleGamma(double shape, double scale) {
        // Marsaglia & Tsang approximation (good enough for MCTS noise)
        if (shape < 1.0) {
            return sampleGamma(1.0 + shape, scale) * Math.pow(m_rnd.nextDouble(), 1.0 / shape);
        }

        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);

        while (true) {
            double x = gaussian();
            double v = 1.0 + c * x;
            if (v <= 0) continue;

            v = v * v * v;
            double u = m_rnd.nextDouble();

            if (u < 1.0 - 0.0331 * x * x * x * x) return d * v * scale;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
        }
    }

    private double gaussian() {
        // Use a true Gaussian for the Gamma sampler.
        return m_rnd.nextGaussian();
    }

    private double[] applyRootDirichlet(double[] priors) {
        if (priors == null) return null;

        double eps = params.DIRICHLET_EPSILON;  // e.g. 0.25
        double alpha = params.DIRICHLET_ALPHA;   // e.g. 0.3

        double[] noise = dirichletNoise(priors.length, alpha);

        double[] out = new double[priors.length];

        for (int i = 0; i < priors.length; i++) {
            out[i] = (1 - eps) * priors[i] + eps * noise[i];
        }

        return out;
    }

}
