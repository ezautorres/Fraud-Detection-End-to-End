# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path
import json
import math

BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"

def _load_json(name: str) -> Any:
    path = ART_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {path}\n"
            f"Ensure you exported artifacts from the notebook into: {ART_DIR}"
        )
    with path.open() as f:
        return json.load(f)

SELECTED_VARS = _load_json("selected_vars.json")          # list[str]
WOE_MAPPINGS  = _load_json("woe_mappings.json")           # {var: {category: woe}}
COEFS         = _load_json("coefficients.json")           # {var: coef}
INTERCEPT     = float(_load_json("intercept.json")["intercept"])
SCORE_PARAMS  = _load_json("score_params.json")           # {PDO, BASE_SCORE, BASE_ODDS, factor, offset, intercept_points}
VERSION_INFO  = (_load_json("version.json")
                 if (ART_DIR / "version.json").exists()
                 else {"model_version": "v1.0"})

MAKE_MAP = {
    "Accura" : "Acura",
    "VW"     : "Volkswagen",
    "Nisson" : "Nissan",
    "Porche" : "Porsche",
    "Mecedes": "Mercedes"
}

def canonicalize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply minimal normalization to reduce common typos/spacing handled in preprocessing.
    Extend here if you need more parity with the notebook.
    """
    out = dict(payload)
    if "Make" in out and isinstance(out["Make"], str):
        out["Make"] = MAKE_MAP.get(out["Make"], out["Make"])
    return out

# Core math.
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def compute_log_odds(client: Dict[str, Any]) -> float:
    """
    Logistic linear predictor using raw categories:
    log_odds = intercept + sum_j coef_j * WOE_j(category_of_client)
    Unknown categories map to WOE = 0.0 (neutral).
    """
    z = INTERCEPT
    for var in SELECTED_VARS:
        raw_val = client.get(var, None)
        w = 0.0
        if raw_val is not None:
            w = float(WOE_MAPPINGS.get(var, {}).get(str(raw_val), 0.0))
        coef = float(COEFS[var])
        z += coef * w
    return z

def compute_score(client: Dict[str, Any]) -> float:
    """
    Points-based score:
    score = offset + intercept_points + sum_j [ - coef_j * WOE_j * factor ]
    """
    factor = float(SCORE_PARAMS["factor"])
    total = float(SCORE_PARAMS["offset"]) + float(SCORE_PARAMS["intercept_points"])
    for var in SELECTED_VARS:
        raw_val = client.get(var, None)
        w = 0.0
        if raw_val is not None:
            w = float(WOE_MAPPINGS.get(var, {}).get(str(raw_val), 0.0))
        coef = float(COEFS[var])
        total += -coef * w * factor
    return round(total, 0)

# ---------- API schema ----------
class ClientPayload(BaseModel):
    data: Dict[str, Any]

# ---------- FastAPI app ----------
app = FastAPI(
    title       = "Fraud Scoring API",
    version     = str(VERSION_INFO.get("model_version", "v1.0")),
    description = "Score fraud risk using a WoE-based logistic scorecard.",
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": VERSION_INFO.get("model_version", "v1.0"),
        "selected_vars": len(SELECTED_VARS),
        "artifacts_dir": str(ART_DIR),
    }

@app.post("/score")
def score(payload: ClientPayload):
    """
    POST /score
    Body:
      {
        "data": {
          "<Variable>": "<raw_category_value>",
          ...
        }
      }
    Notes:
    - Only variables in SELECTED_VARS are used; others are ignored.
    - Unseen/missing categories get WoE=0.0 (neutral).
    """
    client_raw = payload.data
    if not isinstance(client_raw, dict):
        raise HTTPException(status_code = 400, detail = "Invalid payload: 'data' must be an object.")

    client = canonicalize(client_raw)

    # Compute outputs.
    z = compute_log_odds(client)
    p = sigmoid(z)
    s = compute_score(client)

    return {
        "score"             : s,
        "probability_fraud" : round(p, 6),
        "log_odds"          : round(z, 6),
        "used_variables"    : SELECTED_VARS,
        "inputs_used"       : {k: client.get(k, None) for k in SELECTED_VARS},
    }