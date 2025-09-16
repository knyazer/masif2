import matplotlib.pyplot as plt
import numpy as np

# Set a style for the plot
plt.style.use("seaborn-v0_8-whitegrid")

# --- Data Extracted from the Plot ---

# Common x-axis values (Context Length)
x_context = [300, 500, 800, 1600, 3200]

# Data for the Left Plot: Covariance
cov_data = {
    "IFBO baseline": {"y": np.array([4.72, 4.98, 5.23, 5.40, 5.55])},
    "FT 1600": {
        "y": np.array([5.45, 5.58, 5.72, 5.85, 5.91]),
        "err": np.array([0.06, 0.05, 0.04, 0.04, 0.04]),
    },
    "FT 800": {
        "y": np.array([5.07, 5.42, 5.62, 5.73, 5.82]),
        "err": np.array([0.06, 0.05, 0.04, 0.04, 0.03]),
    },
    "FT 100": {
        "y": np.array([5.35, 5.50, 5.65, 5.75, 5.86]),
        "err": np.array([0.06, 0.05, 0.04, 0.04, 0.03]),
    },
    "FT 0": {
        "y": np.array([5.05, 5.40, 5.60, 5.72, 5.81]),
        "err": np.array([0.02, 0.02, 0.02, 0.02, 0.02]),
    },
}

# Data for the Right Plot: Learned
learned_data = {
    "IFBO baseline": {"y": np.array([4.75, 5.00, 5.22, 5.40, 5.55])},
    "FT 1600": {
        "y": np.array([6.10, 6.25, 6.42, 6.50, 6.58]),
        "err": np.array([0.10, 0.08, 0.05, 0.05, 0.04]),
    },
    "FT 800": {
        "y": np.array([5.55, 5.95, 6.22, 6.28, 6.45]),
        "err": np.array([0.10, 0.08, 0.05, 0.05, 0.04]),
    },
    "FT 100": {
        "y": np.array([5.68, 6.05, 6.40, 6.55, 6.65]),
        "err": np.array([0.12, 0.10, 0.08, 0.05, 0.04]),
    },
    "FT 0": {
        "y": np.array([5.05, 5.42, 5.64, 5.85, 5.95]),
        "err": np.array([0.03, 0.02, 0.02, 0.02, 0.02]),
    },
}

# Define colors to match the original plot
colors = {
    "FT 1600": "C0",  # blue
    "FT 800": "C1",  # orange
    "FT 100": "C2",  # green
    "FT 0": "C3",  # red
}

# --- Plotting ---

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# Plot 1: IeFT-PFN (Covariance)
ax1.set_title("IeFT-PFN (Covariance)", fontsize=20, pad=15)
ax1.plot(x_context, cov_data["IFBO baseline"]["y"], "k--", label="IFBO baseline")
for name, color in colors.items():
    ax1.errorbar(
        x_context,
        cov_data[name]["y"],
        yerr=cov_data[name]["err"],
        label=name,
        marker="s",
        capsize=5,
        color=color,
        linewidth=2,
        markersize=7,
    )
ax1.set_xlabel("Context length (tokens)", fontsize=16)
ax1.set_ylabel("MMedLL", fontsize=16)
ax1.tick_params(axis="both", which="major", labelsize=14)
ax1.legend(loc="lower right", ncol=2, fontsize=12)
ax1.set_ylim(4.7, 6.0)


# Plot 2: IeFT-PFN (Learned)
ax2.set_title("IeFT-PFN (Learned)", fontsize=20, pad=15)
ax2.plot(x_context, learned_data["IFBO baseline"]["y"], "k--", label="IFBO baseline")
for name, color in colors.items():
    ax2.errorbar(
        x_context,
        learned_data[name]["y"],
        yerr=learned_data[name]["err"],
        label=name,
        marker="s",
        capsize=5,
        color=color,
        linewidth=2,
        markersize=7,
    )
ax2.set_xlabel("Context length (tokens)", fontsize=16)
# ax2.set_ylabel('MMedLL', fontsize=16) # Y-label is shared, can be omitted
ax2.tick_params(axis="both", which="major", labelsize=14)
ax2.legend(loc="lower right", ncol=2, fontsize=12)
ax2.set_ylim(4.75, 6.75)


# Adjust layout and display the plot
plt.tight_layout(pad=3.0)
plt.show()
plt.savefig("fig.png", dpi=300)
