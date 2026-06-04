"""Generate training curve and comparison plot for slides."""
import json
import matplotlib.pyplot as plt
import numpy as np

# Training curve
with open("training_log.json") as f:
    log = json.load(f)
eps = [x["episode"] for x in log]
rewards = [x["reward"] for x in log]
# Rolling mean
window = 20
rolling = np.convolve(rewards, np.ones(window)/window, mode="valid")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(eps, rewards, alpha=0.3, label="raw")
ax.plot(eps[window-1:], rolling, linewidth=2, label=f"rolling mean (window={window})")
ax.set_xlabel("Episode")
ax.set_ylabel("Total reward")
ax.set_title("Training: REINFORCE policy gradient on Polytopia")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("training_curve.png", dpi=150)
print("Saved training_curve.png")

# Comparison plot
try:
    with open("eval_results.json") as f:
        ev = json.load(f)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    labels = ["Random", "Trained NN"]
    means = [ev["random_mean"], ev["agent_mean"]]
    stds = [np.std(ev["random_scores"]), np.std(ev["agent_scores"])]
    ax2.bar(labels, means, yerr=stds, capsize=8, color=["#888", "#2a8"])
    ax2.set_ylabel("Average score")
    ax2.set_title("Agent performance comparison")
    ax2.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig("comparison.png", dpi=150)
    print("Saved comparison.png")
except FileNotFoundError:
    print("Skipping comparison.png — run evaluate_agent.py first.")