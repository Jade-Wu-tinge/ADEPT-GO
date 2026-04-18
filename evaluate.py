import numpy as np
import torch
import networkx as nx
from sklearn.metrics import average_precision_score


def propagate_go_preds(y_hat: torch.Tensor, goterms, go_graph: nx.DiGraph):
    go2id = {go: i for i, go in enumerate(goterms)}
    for go in goterms:
        if go in go_graph:
            parents = set(goterms).intersection(nx.descendants(go_graph, go))
            gi = go2id[go]
            for p in parents:
                pj = go2id[p]
                y_hat[:, pj] = torch.max(y_hat[:, gi], y_hat[:, pj])
    return y_hat

def evaluate_cafa_short(y_true_np: np.ndarray, y_score_np: np.ndarray, goterms, go_graph, ont="mf", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Yt = torch.tensor(y_true_np, dtype=torch.float64, device=device)
    Ys = torch.tensor(y_score_np, dtype=torch.float64, device=device)

    Ys = propagate_go_preds(Ys, goterms, go_graph)

    keep = torch.where(Yt.sum(0) > 0)[0]
    micro = average_precision_score(Yt[:, keep].cpu().numpy(), Ys[:, keep].cpu().numpy(), average="micro")
    macro = average_precision_score(Yt[:, keep].cpu().numpy(), Ys[:, keep].cpu().numpy(), average="macro")

    n = Yt.shape[0]
    goterms_np = np.asarray(goterms)
    ont2root = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}
    root = ont2root[ont]

    y_true_arr = Yt.cpu().numpy()
    prot2true = {}
    for i in range(n):
        all_gos = set()
        for go in goterms_np[np.where(y_true_arr[i] == 1)[0]]:
            all_gos = all_gos.union(nx.descendants(go_graph, go))
            all_gos.add(go)
        all_gos.discard(root)
        prot2true[i] = all_gos

    F_list, AvgPr_list, AvgRc_list = [], [], []
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (Ys > threshold).int().cpu().numpy()
        m = 0
        precision = 0.0
        recall = 0.0
        for i in range(n):
            pred_gos = set()
            for go in goterms_np[np.where(predictions[i] == 1)[0]]:
                pred_gos = pred_gos.union(nx.descendants(go_graph, go))
                pred_gos.add(go)
            pred_gos.discard(root)

            num_pred = len(pred_gos)
            num_true = len(prot2true[i])
            num_overlap = len(prot2true[i].intersection(pred_gos))
            if num_pred > 0 and num_true > 0:
                m += 1
                precision += float(num_overlap) / num_pred
                recall += float(num_overlap) / num_true

        if m > 0:
            avg_pr = precision / m
            avg_rc = recall / n
            if avg_pr + avg_rc > 0:
                f_score = 2 * (avg_pr * avg_rc) / (avg_pr + avg_rc)
                F_list.append(f_score)
                AvgPr_list.append(avg_pr)
                AvgRc_list.append(avg_rc)

    F = np.asarray(F_list)
    R = np.asarray(AvgRc_list)
    P = np.asarray(AvgPr_list)
    fmax = float(np.max(F)) if F.size > 0 else float("nan")

    if R.size > 0:
        order = np.argsort(R)
        aupr_cafa = float(np.trapz(P[order], R[order]))
    else:
        aupr_cafa = float("nan")

    return {
        "Fmax": fmax,
        "AUPR_macro": float(macro),
        "AUPR_micro": float(micro),
        "AUPR_CAFA_protein": aupr_cafa,
    }
