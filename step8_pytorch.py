"""
STEP 8 - A neural network with PyTorch: can it beat the logistic regression of step 5?

Same question as step 5: on day t, will ocean water flow in on day t+2?
Same 24 inputs (from features.py), same split by time:
    train 2000-2016 | validation 2017-2019 | test 2020-2025
So the comparison with step 5 is fair.

The network ("multi-layer perceptron"):
    24 inputs -> 32 neurons -> 16 neurons -> 1 output (the inflow probability)
Unlike logistic regression it can learn COMBINATIONS, e.g. "north-west wind is only
dangerous when the tide is also high".

Needs output/daily_dataset.csv (step 3) and output/model.joblib (step 5).
One-time install:  python -m pip install torch
"""
# %% Cell 1: imports
import copy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from features import build_features, build_target
from helpers import choose_threshold, evaluate_rule

BLUE, ORANGE, INK_2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9"


# %% Cell 2: the same data and split as step 5
data = pd.read_csv("output/daily_dataset.csv", index_col="date", parse_dates=True)
X = build_features(data)
y = build_target(data)
ok = X.notna().all(axis=1) & y.notna()
X, y = X[ok], y[ok].astype(int)

train = X.index < "2017-01-01"
valid = (X.index >= "2017-01-01") & (X.index < "2020-01-01")
test = X.index >= "2020-01-01"

# Neural networks learn best when every input has mean 0 and spread 1.
# Mean and spread come from the TRAINING years only (no peeking at the test years).
mean, std = X[train].mean(), X[train].std()


def to_tensor(part):
    """Pandas rows -> a PyTorch tensor of scaled inputs."""
    return torch.tensor(((X[part] - mean) / std).values, dtype=torch.float32)


X_train, X_valid, X_test = to_tensor(train), to_tensor(valid), to_tensor(test)
y_train = torch.tensor(y[train].values, dtype=torch.float32)
print(f"train {len(X_train)} days, validation {len(X_valid)}, test {len(X_test)}, {X.shape[1]} inputs")


# %% Cell 3: the network
def make_network(n_inputs):
    return nn.Sequential(
        nn.Linear(n_inputs, 32), nn.ReLU(), nn.Dropout(0.2),  # dropout: randomly switch off
        nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),        # neurons while training, so the
        nn.Linear(16, 1),                                     # network can't memorise days
    )


def probabilities(net, inputs):
    """Network output -> inflow probability between 0 and 1."""
    net.eval()  # switch dropout off
    with torch.no_grad():
        return torch.sigmoid(net(inputs)).squeeze(1).numpy()


# %% Cell 4: training
# Inflow is rare (~5% of days), so an inflow day counts ~19x more in the loss,
# the same idea as class_weight="balanced" in step 5.
# We stop when the validation score hasn't improved for 20 rounds ("early stopping")
# and keep the best version: training longer would only memorise the training years.
def train_network(seed, epochs=300, patience=20):
    torch.manual_seed(seed)
    net = make_network(X_train.shape[1])
    weight = (y_train == 0).sum() / (y_train == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    best_score, best_net, waited, history = -1, None, 0, []
    for epoch in range(epochs):
        net.train()
        losses = []
        for batch in torch.randperm(len(X_train)).split(128):  # 128 random days at a time
            optimizer.zero_grad()
            loss = loss_fn(net(X_train[batch]).squeeze(1), y_train[batch])
            loss.backward()   # work out how each weight should change
            optimizer.step()  # change the weights a little bit
            losses.append(loss.item())
        score = average_precision_score(y[valid], probabilities(net, X_valid))
        history.append({"epoch": epoch + 1, "loss": np.mean(losses), "valid_score": score})
        if score > best_score:
            best_score, best_net, waited = score, copy.deepcopy(net), 0
        else:
            waited += 1
            if waited >= patience:
                break
    return best_net, pd.DataFrame(history)


net, history = train_network(seed=0)
best = history.loc[history["valid_score"].idxmax()]
print(f"stopped after {len(history)} rounds; best validation score {best['valid_score']:.2f} "
      f"in round {best['epoch']:.0f}")


# %% Cell 5: CHART - how training went
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
ax1.plot(history["epoch"], history["loss"], color=BLUE)
ax1.set_title("Training loss (lower = fits training years better)", loc="left", fontweight="bold")
ax2.plot(history["epoch"], history["valid_score"], color=ORANGE)
ax2.axvline(best["epoch"], color=INK_2, linestyle="--", linewidth=1)
ax2.text(best["epoch"], history["valid_score"].min(), "  best: kept", color=INK_2, fontsize=8)
ax2.set_title("Validation score 2017–2019 (higher = better)", loc="left", fontweight="bold")
for ax in (ax1, ax2):
    ax.set_xlabel("training round (epoch)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID)
fig.tight_layout()
fig.savefig("output/step8_training_curve.png", dpi=150)
plt.show()


# %% Cell 6: is it luck? Train 5 networks with different random starts
# A network starts from random weights, so every run is a bit different.
# If the result changes a lot between runs, one good run proves nothing.
runs = [train_network(seed)[0] for seed in range(5)]
test_scores = [average_precision_score(y[test], probabilities(n, X_test)) for n in runs]
print("test score of 5 runs:", ", ".join(f"{s:.2f}" for s in test_scores),
      f"-> average {np.mean(test_scores):.2f}")


# %% Cell 7: neural network vs logistic regression on the TEST years (2020-2025)
step5 = joblib.load("output/model.joblib")
logistic = step5["model"].predict_proba(X[step5["features"]])[:, 1]
network = np.mean([probabilities(n, to_tensor(slice(None))) for n in runs], axis=0)  # average of 5 runs


def as_rule_input(prob, period):
    part = pd.DataFrame({"prob": prob, "inflow_t+2": y}, index=X.index)[period]
    part.index = part.index + pd.Timedelta(days=2)  # the day the prediction is ABOUT
    return part


rows = {}
for name, prob in [("logistic regression (step 5)", logistic), ("neural network (PyTorch)", network),
                   ("average of both", (logistic + network) / 2)]:
    threshold = choose_threshold(as_rule_input(prob, valid))  # same rule as step 6
    result = evaluate_rule(as_rule_input(prob, test), threshold)
    rows[name] = {"score (avg precision)": average_precision_score(y[test], prob[test]),
                  "threshold": threshold, "events caught": result["share_caught"],
                  "alarms right": result["alarm_right"],
                  "unnecessary pump days/yr": result["unnecessary_per_year"]}
comparison = pd.DataFrame(rows).T
print("TEST period 2020-2025:")
print(comparison.round(2).to_string())
comparison.round(3).to_csv("output/step8_comparison.csv")


# %% Cell 8: save the networks (predict.py still uses step 5's model, see the comparison)
torch.save({"state_dicts": [n.state_dict() for n in runs], "features": list(X.columns),
            "mean": mean.to_dict(), "std": std.to_dict(),
            "threshold": float(comparison.loc["neural network (PyTorch)", "threshold"])},
           "output/nn_model.pt")
print("saved output/nn_model.pt")
