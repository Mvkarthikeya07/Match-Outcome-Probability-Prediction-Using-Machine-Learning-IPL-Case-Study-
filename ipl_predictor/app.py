# app.py
import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from math import exp

app = Flask(__name__)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "encoders.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENC_PATH):
    raise FileNotFoundError("Model or encoders not found in models/. Run train_model.py first.")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENC_PATH)
label_enc = encoders["label"]

# populate options for UI
teams = sorted(list(encoders["team1"].classes_)) if "team1" in encoders else []
cities = sorted(list(encoders["city"].classes_)) if "city" in encoders else []

def encode_match_features(team1, team2, toss_winner, toss_decision, city):
    """Encode categorical fields into model input. If unseen value, fallback to 0."""
    row = {"team1": team1, "team2": team2, "toss_winner": toss_winner,
           "toss_decision": toss_decision, "city": city}
    df = pd.DataFrame([row])
    # toss_decision numeric
    df["toss_decision"] = df["toss_decision"].astype(str).str.lower().map({"bat":1,"field":0}).fillna(0).astype(int)
    for col in ["team1","team2","toss_winner","city"]:
        le = encoders.get(col)
        if le is not None:
            try:
                df[col] = le.transform(df[col].astype(str))
            except Exception:
                df[col] = 0
        else:
            df[col] = 0
    return df[["team1","team2","toss_winner","toss_decision","city"]]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def match_state_adjustment(batting_team, bowling_team, score, overs, wickets, target):
    """
    Heuristic that returns an adjustment score in range [-1, +1] where positive favors batting team.
    - If target <= 0, we assume batting is defending (first innings) -> small effect.
    - If target > 0, compute required run-rate vs current run-rate and factor wickets remaining.
    """
    # Basic guardrails
    if overs <= 0:
        current_rr = 0.0
    else:
        current_rr = score / overs

    # Overs remaining in T20 (assume 20)
    max_overs = 20.0
    overs_left = max(max_overs - overs, 0.0001)

    # If chasing:
    if target and target > 0:
        runs_remaining = max(target - score, 0)
        if overs_left <= 0:
            req_rr = 999.0 if runs_remaining > 0 else 0.0
        else:
            req_rr = runs_remaining / overs_left

        # rr_ratio: >1 means chasing is behind required
        if req_rr <= 0:
            rr_ratio = 2.0
        else:
            rr_ratio = (current_rr + 1e-6) / (req_rr + 1e-6)

        # wickets impact: more remaining wickets -> more chance for batting
        wickets_left = max(10 - wickets, 0)
        wicket_factor = (wickets_left / 10.0)  # 0..1

        # Build an adjustment magnitude using rr_ratio and wicket factor.
        # If rr_ratio >>1 then batting is cruising -> positive. If rr_ratio <<1 then batting struggling -> negative.
        rr_score = (rr_ratio - 1.0) * 2.0  # scale so small differences matter
        # convert rr_score to -4..+4 roughly, pass through sigmoid and center
        adj_raw = sigmoid(rr_score) - 0.5  # in -0.5..+0.5
        # scale by wicket_factor and by a tunable weight (0.9)
        adj = adj_raw * 2.0 * wicket_factor  # range approx -1..+1
        return float(adj)
    else:
        # First innings case: rely mainly on score vs expected par (simple heuristic)
        # Expectation: average 160 in T20; if score much above, batting-team more likely (defend)
        par = 160.0
        diff = (score - par) / par  # e.g. 0.1 if 176
        # weigh by wickets left
        wickets_left = max(10 - wickets, 0)
        wicket_factor = (wickets_left / 10.0)
        adj = diff * 0.6 * wicket_factor  # smaller effect
        # bound
        adj = max(min(adj, 0.9), -0.9)
        return float(adj)

def combine_probs(base_batting, base_bowling, adj, weight_model=0.7):
    """
    Combine base model probability and match-state adjustment.
    weight_model is how much weight to give the model base probability (0..1).
    adj in range roughly [-1, +1] indicating advantage for batting side.
    We map adj to a delta in probability range [-0.45, +0.45] for knobs.
    """
    delta = max(min(adj, 1.0), -1.0) * 0.45
    # base_batting is e.g. 0.55, base_bowling = 0.45
    combined_batting = weight_model * base_batting + (1 - weight_model) * (0.5 + delta)
    combined_bowling = 1.0 - combined_batting
    # final clamp
    combined_batting = max(0.0, min(1.0, combined_batting))
    combined_bowling = max(0.0, min(1.0, combined_bowling))
    return combined_batting, combined_bowling

@app.route("/")
def index():
    return render_template("index.html", teams=teams, cities=cities)

@app.route("/predict", methods=["POST"])
def predict():
    # Inputs
    batting_team = request.form.get("batting_team")
    bowling_team = request.form.get("bowling_team")
    city = request.form.get("city") or ""
    toss_winner = request.form.get("toss_winner") or batting_team
    toss_decision = request.form.get("toss_decision") or "field"
    # numeric inputs (try-cast)
    try:
        target = float(request.form.get("target") or 0.0)
    except:
        target = 0.0
    try:
        score = float(request.form.get("score") or 0.0)
    except:
        score = 0.0
    try:
        overs = float(request.form.get("overs") or 0.0)
    except:
        overs = 0.0
    try:
        wickets = int(float(request.form.get("wickets") or 0))
    except:
        wickets = 0

    # If same teams -> show error
    if batting_team == bowling_team:
        return render_template("index.html", teams=teams, cities=cities, error="Batting and Bowling teams cannot be the same.")

    # Build base model input. Note: model was trained with 'team1' and 'team2' as playing teams; we'll set
    # team1=batting_team, team2=bowling_team for the model input (consistent with training convention)
    X = encode_match_features(batting_team, bowling_team, toss_winner, toss_decision, city)
    # get model probabilities across all classes
    try:
        proba_all = model.predict_proba(X)[0]  # shape (n_classes,)
    except Exception:
        # If model doesn't support predict_proba (unlikely), fallback to predict and make it artificial
        pred_idx = model.predict(X)[0]
        proba_all = np.zeros(len(label_enc.classes_))
        proba_all[pred_idx] = 1.0

    # Map probabilities to names
    classes = label_enc.inverse_transform(np.arange(len(label_enc.classes_)))
    proba_map = dict(zip(classes, proba_all))

    # Base probabilities for the two teams
    base_batting = float(proba_map.get(batting_team, 0.5))
    base_bowling = float(proba_map.get(bowling_team, 0.5))
    # normalize if both small or missing
    s = base_batting + base_bowling
    if s == 0:
        base_batting, base_bowling = 0.5, 0.5
    else:
        base_batting, base_bowling = base_batting / s, base_bowling / s

    # Match-state adjustment (heuristic)
    adj = match_state_adjustment(batting_team, bowling_team, score, overs, wickets, target)

    # Combine
    final_batting, final_bowling = combine_probs(base_batting, base_bowling, adj, weight_model=0.7)

    # Create top-two ordered list for display
    results = [
        {"team": batting_team, "prob": round(final_batting * 100, 2)},
        {"team": bowling_team, "prob": round(final_bowling * 100, 2)}
    ]

    # Ensure sum is 100 with small rounding fix
    s100 = results[0]["prob"] + results[1]["prob"]
    if s100 != 100.0:
        diff = 100.0 - s100
        # add diff to higher probability team
        if results[0]["prob"] >= results[1]["prob"]:
            results[0]["prob"] = round(results[0]["prob"] + diff, 2)
        else:
            results[1]["prob"] = round(results[1]["prob"] + diff, 2)

    return render_template("index.html", teams=teams, cities=cities,
                           prediction=results, inputs={
                               "batting_team": batting_team,
                               "bowling_team": bowling_team,
                               "city": city,
                               "toss_winner": toss_winner,
                               "toss_decision": toss_decision,
                               "target": target,
                               "score": score,
                               "overs": overs,
                               "wickets": wickets
                           })

if __name__ == "__main__":
    app.run(debug=True)
