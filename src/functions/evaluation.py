import plotly.graph_objects as go


def plot_metric(results, metric="val_loss", title=None):
    """One line per run. metric is any column of the history: train_loss, val_loss, train_acc, val_acc, grad_norm, ..."""
    fig = go.Figure()
    for label, r in results.items():
        fig.add_trace(go.Scatter(x=r["history"].index, y=r["history"][metric], mode="lines+markers", name=label))
    fig.update_layout(template="plotly_dark", title=title or metric, xaxis_title="Epoch", yaxis_title=metric)
    fig.show()


def summary(results):
    """Final-epoch numbers for every run, side by side. loss_gap = val_loss - train_loss (bigger = more overfitting)."""
    rows = {}
    for label, r in results.items():
        last = r["history"].iloc[-1]
        rows[label] = {
            "train_loss": last["train_loss"], "val_loss": last["val_loss"],
            "train_acc": last["train_acc"], "val_acc": last["val_acc"],
            "loss_gap": last["val_loss"] - last["train_loss"],
            **r["weights"],
            "seconds": r["seconds"],
        }
    return pd.DataFrame(rows).T.round(4)


def test_score(result):
    """Loss and accuracy on the held-out MNIST test set. Use once, on your final chosen model."""
    return evaluate(result["params"], X_test, y_test, result["config"]["activation"])