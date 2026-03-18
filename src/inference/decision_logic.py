"""
Rule-based decision engine for Plant Watering System.

Implements a weighted rule system for plant health classification
and watering recommendations.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import (
    RULE_THRESHOLDS,
    merge_rule_thresholds,
    CLASS_NAMES,
    CLASS_ICONS,
    CLASS_COLORS,
)


def _build_rule_map(thresholds):
    """Build the rule map using the current threshold values."""
    return [
        # Needs-Water rules (vote -> 1)
        {
            "id": "NW-01",
            "label": "Low Soil Moisture",
            "vote": 1,
            "weight": 3,
            "condition": lambda r: r.get("Soil_Moisture", 50) < thresholds["soil_moisture_low"],
            "reason": f"Soil moisture below {thresholds['soil_moisture_low']}% -> plant needs water",
        },
        {
            "id": "NW-02",
            "label": "Days Since Last Watering",
            "vote": 1,
            "weight": 2,
            "condition": lambda r: r.get("days_since_last_watering", 0) > thresholds["days_since_water"],
            "reason": f"Plant not watered for >{thresholds['days_since_water']} days",
        },
        {
            "id": "NW-03",
            "label": "High Temperature Stress",
            "vote": 1,
            "weight": 1,
            "condition": lambda r: r.get("Ambient_Temperature", 25) > thresholds["temperature_high"],
            "reason": f"Ambient temperature >{thresholds['temperature_high']}C increases water demand",
        },
        {
            "id": "NW-04",
            "label": "Low Humidity",
            "vote": 1,
            "weight": 1,
            "condition": lambda r: r.get("Humidity", 60) < thresholds["humidity_low"],
            "reason": f"Humidity below {thresholds['humidity_low']}% -> higher evaporation rate",
        },

        # Overwatered rules (vote -> 2)
        {
            "id": "OW-01",
            "label": "High Soil Moisture",
            "vote": 2,
            "weight": 3,
            "condition": lambda r: r.get("Soil_Moisture", 50) > thresholds["soil_moisture_high"],
            "reason": f"Soil moisture above {thresholds['soil_moisture_high']}% -> overwatering risk",
        },
        {
            "id": "OW-02",
            "label": "Nitrogen Excess",
            "vote": 2,
            "weight": 1,
            "condition": lambda r: r.get("Nitrogen_Level", 50) > 80,
            "reason": "High nitrogen often co-occurs with overwatering",
        },

        # Healthy rules (vote -> 0)
        {
            "id": "HE-01",
            "label": "Optimal Soil Moisture",
            "vote": 0,
            "weight": 3,
            "condition": lambda r: thresholds["soil_moisture_low"] <= r.get("Soil_Moisture", 50) <= thresholds["soil_moisture_high"],
            "reason": f"Soil moisture in optimal range [{thresholds['soil_moisture_low']} - {thresholds['soil_moisture_high']}%]",
        },
        {
            "id": "HE-02",
            "label": "Normal Temperature",
            "vote": 0,
            "weight": 1,
            "condition": lambda r: r.get("Ambient_Temperature", 25) <= thresholds["temperature_high"],
            "reason": "Temperature within acceptable range",
        },
        {
            "id": "HE-03",
            "label": "Adequate Humidity",
            "vote": 0,
            "weight": 1,
            "condition": lambda r: r.get("Humidity", 60) >= thresholds["humidity_low"],
            "reason": "Humidity sufficient for healthy growth",
        },
    ]


def rule_based_decision(sensor_reading: dict, thresholds: dict | None = None) -> dict:
    """
    Apply rule-based decision logic to sensor readings.

    Args:
        sensor_reading: Dictionary with sensor feature values.
        thresholds: Optional threshold override dictionary.

    Returns:
        dict: Decision result with predicted class, confidence, and triggered rules.
    """
    active_thresholds = merge_rule_thresholds(thresholds or RULE_THRESHOLDS)
    rule_map = _build_rule_map(active_thresholds)

    scores = {0: 0.0, 1: 0.0, 2: 0.0}
    triggered = []
    skipped = []

    for rule in rule_map:
        try:
            fired = rule["condition"](sensor_reading)
        except Exception:
            fired = False

        if fired:
            scores[rule["vote"]] += rule["weight"]
            triggered.append({
                "id": rule["id"],
                "label": rule["label"],
                "reason": rule["reason"],
                "weight": rule["weight"],
                "vote": rule["vote"],
            })
        else:
            skipped.append({"id": rule["id"], "label": rule["label"]})

    # Determine winner
    predicted_class = max(scores, key=scores.__getitem__)

    total = sum(scores.values()) or 1
    confidence = scores[predicted_class] / total

    return {
        "predicted_class": predicted_class,
        "label": CLASS_NAMES[predicted_class],
        "icon": CLASS_ICONS[predicted_class],
        "color": CLASS_COLORS[predicted_class],
        "confidence": round(confidence, 3),
        "scores": scores,
        "thresholds": active_thresholds,
        "triggered_rules": triggered,
        "skipped_rules": skipped,
    }


def get_watering_action(decision: dict) -> dict:
    """
    Convert a rule-based decision into a concrete watering action.

    Args:
        decision: Output from rule_based_decision().

    Returns:
        dict: Watering action with pump status, duration, and advice.
    """
    cls = decision["predicted_class"]

    if cls == 1:  # Needs Water
        return {
            "action": "WATER NOW",
            "pump_on": True,
            "duration_minutes": 5,
            "urgency": "HIGH",
            "advice": "Turn on pump immediately. Water for ~5 minutes.",
        }
    elif cls == 2:  # Overwatered
        return {
            "action": "STOP WATERING",
            "pump_on": False,
            "duration_minutes": 0,
            "urgency": "MEDIUM",
            "advice": "Do NOT water. Allow soil to dry. Check drainage.",
        }
    else:  # Healthy
        return {
            "action": "NO ACTION",
            "pump_on": False,
            "duration_minutes": 0,
            "urgency": "LOW",
            "advice": "Plant is healthy. Continue regular monitoring.",
        }


if __name__ == "__main__":
    # CLI test
    sample = {
        "Soil_Moisture": 22,
        "Ambient_Temperature": 36,
        "Humidity": 30,
        "Nitrogen_Level": 40,
        "days_since_last_watering": 4,
    }
    result = rule_based_decision(sample)
    action = get_watering_action(result)

    print(f"\n{'='*50}")
    print(f"  Decision: {result['icon']} {result['label']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Scores: {result['scores']}")
    print(f"  Action: {action['action']} (Pump: {action['pump_on']})")
    print(f"  Advice: {action['advice']}")
    print(f"  Triggered rules ({len(result['triggered_rules'])}):")
    for r in result['triggered_rules']:
        print(f"    [{r['id']}] {r['label']} (weight={r['weight']})")
    print(f"{'='*50}\n")
