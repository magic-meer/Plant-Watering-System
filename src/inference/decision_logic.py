"""
decision_logic.py  ─  Rule-based decision engine for Plant Watering System
=============================================================================
Step 1: Define rule mapping
Step 2: Implement decision function
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import RULE_THRESHOLDS, CLASS_NAMES, CLASS_ICONS, CLASS_COLORS


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RULE MAPPING
#     Each rule is a dict with: name, condition (callable), weight, reason
# ══════════════════════════════════════════════════════════════════════════════

def _t(key):
    """Shortcut to get threshold value."""
    return RULE_THRESHOLDS[key]


RULE_MAP = [
    # ── Needs-Water rules (vote → 1) ─────────────────────────────────────
    {
        "id":        "NW-01",
        "label":     "Low Soil Moisture",
        "vote":      1,
        "weight":    3,
        "condition": lambda r: r.get("Soil_Moisture", 50) < _t("soil_moisture_low"),
        "reason":    f"Soil moisture below {_t('soil_moisture_low')}% → plant needs water",
    },
    {
        "id":        "NW-02",
        "label":     "Days Since Last Watering",
        "vote":      1,
        "weight":    2,
        "condition": lambda r: r.get("days_since_last_watering", 0) > _t("days_since_water"),
        "reason":    f"Plant not watered for >{_t('days_since_water')} days",
    },
    {
        "id":        "NW-03",
        "label":     "High Temperature Stress",
        "vote":      1,
        "weight":    1,
        "condition": lambda r: r.get("Ambient_Temperature", 25) > _t("temperature_high"),
        "reason":    f"Ambient temperature >{_t('temperature_high')}°C increases water demand",
    },
    {
        "id":        "NW-04",
        "label":     "Low Humidity",
        "vote":      1,
        "weight":    1,
        "condition": lambda r: r.get("Humidity", 60) < _t("humidity_low"),
        "reason":    f"Humidity below {_t('humidity_low')}% → higher evaporation rate",
    },

    # ── Overwatered rules (vote → 2) ──────────────────────────────────────
    {
        "id":        "OW-01",
        "label":     "High Soil Moisture",
        "vote":      2,
        "weight":    3,
        "condition": lambda r: r.get("Soil_Moisture", 50) > _t("soil_moisture_high"),
        "reason":    f"Soil moisture above {_t('soil_moisture_high')}% → overwatering risk",
    },
    {
        "id":        "OW-02",
        "label":     "Nitrogen Excess",
        "vote":      2,
        "weight":    1,
        "condition": lambda r: r.get("Nitrogen_Level", 50) > 80,
        "reason":    "High nitrogen often co-occurs with overwatering",
    },

    # ── Healthy rules (vote → 0) ───────────────────────────────────────────
    {
        "id":        "HE-01",
        "label":     "Optimal Soil Moisture",
        "vote":      0,
        "weight":    3,
        "condition": lambda r: _t("soil_moisture_low") <= r.get("Soil_Moisture", 50) <= _t("soil_moisture_high"),
        "reason":    f"Soil moisture in optimal range [{_t('soil_moisture_low')}–{_t('soil_moisture_high')}%]",
    },
    {
        "id":        "HE-02",
        "label":     "Normal Temperature",
        "vote":      0,
        "weight":    1,
        "condition": lambda r: r.get("Ambient_Temperature", 25) <= _t("temperature_high"),
        "reason":    "Temperature within acceptable range",
    },
    {
        "id":        "HE-03",
        "label":     "Adequate Humidity",
        "vote":      0,
        "weight":    1,
        "condition": lambda r: r.get("Humidity", 60) >= _t("humidity_low"),
        "reason":    "Humidity sufficient for healthy growth",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DECISION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def rule_based_decision(sensor_reading: dict) -> dict:
    """
    Apply RULE_MAP to a sensor reading dict and return a decision.

    Parameters
    ----------
    sensor_reading : dict
        Keys match feature names, e.g.
        {"Soil_Moisture": 25, "Ambient_Temperature": 30, ...}

    Returns
    -------
    dict with keys:
        predicted_class   int        (0 / 1 / 2)
        label             str        ("Healthy" / "Needs Water" / "Overwatered")
        icon              str        emoji
        color             str        hex color
        confidence        float      weighted score ratio
        triggered_rules   list[dict] rules that fired
        skipped_rules     list[dict] rules that did NOT fire
        scores            dict       {0: score, 1: score, 2: score}
    """
    scores         = {0: 0.0, 1: 0.0, 2: 0.0}
    triggered      = []
    skipped        = []

    for rule in RULE_MAP:
        try:
            fired = rule["condition"](sensor_reading)
        except Exception:
            fired = False

        if fired:
            scores[rule["vote"]] += rule["weight"]
            triggered.append({
                "id":     rule["id"],
                "label":  rule["label"],
                "reason": rule["reason"],
                "weight": rule["weight"],
                "vote":   rule["vote"],
            })
        else:
            skipped.append({"id": rule["id"], "label": rule["label"]})

    # ── Winner ────────────────────────────────────────────────────────────
    predicted_class = max(scores, key=scores.__getitem__)

    total = sum(scores.values()) or 1
    confidence = scores[predicted_class] / total

    return {
        "predicted_class": predicted_class,
        "label":           CLASS_NAMES[predicted_class],
        "icon":            CLASS_ICONS[predicted_class],
        "color":           CLASS_COLORS[predicted_class],
        "confidence":      round(confidence, 3),
        "scores":          scores,
        "triggered_rules": triggered,
        "skipped_rules":   skipped,
    }


def get_watering_action(decision: dict) -> dict:
    """
    Convert a decision dict into a concrete watering action.

    Returns
    -------
    dict with keys: action, pump_on, duration_minutes, urgency, advice
    """
    cls = decision["predicted_class"]

    if cls == 1:          # Needs Water
        return {
            "action":           "WATER NOW",
            "pump_on":          True,
            "duration_minutes": 5,
            "urgency":          "HIGH",
            "advice":           "Turn on pump immediately. Water for ~5 minutes.",
        }
    elif cls == 2:        # Overwatered
        return {
            "action":           "STOP WATERING",
            "pump_on":          False,
            "duration_minutes": 0,
            "urgency":          "MEDIUM",
            "advice":           "Do NOT water. Allow soil to dry. Check drainage.",
        }
    else:                 # Healthy
        return {
            "action":           "NO ACTION",
            "pump_on":          False,
            "duration_minutes": 0,
            "urgency":          "LOW",
            "advice":           "Plant is healthy. Continue regular monitoring.",
        }


# ── Quick CLI test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "Soil_Moisture":          22,
        "Ambient_Temperature":    36,
        "Humidity":               30,
        "Nitrogen_Level":         40,
        "days_since_last_watering": 4,
    }
    result = rule_based_decision(sample)
    action = get_watering_action(result)

    print(f"\n{'='*50}")
    print(f"  Decision : {result['icon']} {result['label']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Scores   : {result['scores']}")
    print(f"  Action   : {action['action']} (Pump: {action['pump_on']})")
    print(f"  Advice   : {action['advice']}")
    print(f"  Triggered rules ({len(result['triggered_rules'])}):")
    for r in result['triggered_rules']:
        print(f"    [{r['id']}] {r['label']} (weight={r['weight']})")
    print(f"{'='*50}\n")
