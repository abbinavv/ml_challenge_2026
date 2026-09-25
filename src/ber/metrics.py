"""The official evaluation metric: macro-averaged F0.5 over Source-1 entities.

Per entity:
  * no true matches  -> 1.0 if the prediction is empty, else 0.0 (singleton rule)
  * true matches     -> F0.5 of the predicted set (0.0 if nothing correct is predicted)
The score is the plain mean over all evaluated entities.
"""

BETA2 = 0.25  # beta = 0.5


def f05(pred, true):
    """F0.5 for one entity given predicted and true ID sets."""
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(preds, truths, entities):
    """Mean F0.5 over `entities`; preds/truths map entity -> iterable of IDs."""
    total = 0.0
    for e in entities:
        total += f05(preds.get(e, ()), truths.get(e, ()))
    return total / len(entities) if entities else 0.0


def report(preds, truths, entities):
    """Macro F0.5 plus pair-level precision/recall and the singleton-only score."""
    tp = fp = fn = 0
    single, single_n = 0.0, 0
    for e in entities:
        p, t = set(preds.get(e, ())), set(truths.get(e, ()))
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
        if not t:
            single_n += 1
            single += 1.0 if not p else 0.0
    return {
        "macro_f05": macro_f05(preds, truths, entities),
        "pair_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pair_recall": tp / (tp + fn) if tp + fn else 0.0,
        "singleton_score": single / single_n if single_n else float("nan"),
        "entities": len(entities),
    }


if __name__ == "__main__":
    # Worked example from the problem statement: predicted 3, true 2 (both correct).
    got = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(got - 0.714) < 1e-3, got
    assert f05(set(), set()) == 1.0 and f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0
    print(f"metric self-test passed (example = {got:.3f})")
