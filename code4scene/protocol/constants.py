"""Frozen constants of the Code4Scene paper scoring protocol.

Every number here appears in the paper (Section 2.4 and Appendix C). Changing
any of them defines a different protocol; the policy identifiers below are
recorded in every score this package produces so results remain traceable.
"""

from __future__ import annotations

#: Case score for image-to-scene (paper Eq. 6): 0.8 * Repair F1 + 0.2 * Physics.
I2S_CASE_POLICY = "i2s-actor-f1-0.8-physics-0.2-case-macro.v1"
#: Actor-level Repair F1 with ID-independent input/candidate correspondence.
ACTOR_F1_POLICY = "unified-actor-repair-success-f1.v2"
#: Input/candidate correspondence used by the Actor F1 metric.
CORRESPONDENCE_POLICY = "semantic-input-candidate-correspondence.v1"
#: Physical Safety: two fixed-weight leaves; unavailable or N/A leaves score 0.
PHYSICS_POLICY = "physics-all-leaves-zero.v2"
#: Text-to-scene support rule (0-5 cm support gap, 5 cm ground tolerance).
T2S_FLOATING_METRIC = "t2s-aabb-floating-contact-5cm.v2"
#: Text-to-scene case score: 0.20 Detailed + 0.60 Overview + 0.20 Physics.
T2S_CASE_POLICY = "text-to-scene-human-aligned"
#: Model score (paper Eq. 7): 0.5 * S_T2S + 0.5 * S_I2S, with S_I2S the mean of
#: all image-to-scene cases (indoor and outdoor cases weighted equally).
MODEL_POLICY = "t2s-0.5-i2s-0.5-pooled-case-macro.v2"

# --- Physical Safety (Appendix C.3) -------------------------------------
PHYSICS_LEAF_WEIGHTS = {"floating": 0.5, "solid_penetration": 0.5}
PHYSICS_DECIMALS = 4
SUPPORT_GAP_CM = 5.0

# --- Text-to-scene (Appendix C.4, C.6, Table 9) --------------------------
T2S_WEIGHTS = {"detailed": 0.20, "overview": 0.60, "physics": 0.20}
T2S_DECIMALS = 4
DETAILED_FAMILY_WEIGHTS = {
    "identity_environment": 0.25,
    "content_quantity": 0.40,
    "attributes_materials": 0.15,
    "spatial_composition": 0.20,
}
DETAILED_DECIMALS = 4
OVERVIEW_DIMENSION_WEIGHTS = {
    "global_prompt_alignment": 0.40,
    "composition_and_layout": 0.25,
    "style_atmosphere_coherence": 0.20,
    "completeness_and_polish": 0.15,
}
OVERVIEW_STRUCTURAL_FLOOR = 0.75
OVERVIEW_STRUCTURAL_WEIGHT = 0.25
OVERVIEW_SEVERE_CAP = 0.40
OVERVIEW_SEVERE_MIN_CONFIDENCE = 0.80
OVERVIEW_SEVERE_MIN_VIEWS = 2
OVERVIEW_DECIMALS = 4

# --- Image-to-scene (Appendix C.5, C.6) ---------------------------------
I2S_REPAIR_WEIGHT = 0.80
I2S_PHYSICS_WEIGHT = 0.20
I2S_DECIMALS = 6
#: Nominal repair-success tolerances (Table 7).
REPAIR_POSITION_CM = 5.0
REPAIR_ROTATION_DEG = 5.0
REPAIR_RELATIVE_SCALE = 0.05

# --- Model score (Appendix C.6) -----------------------------------------
MODEL_WEIGHTS = {"t2s": 0.5, "i2s": 0.5}

#: Settings of the benchmark, as named in task files and evidence bundles.
SETTING_T2S = "text-to-scene"
SETTING_INDOOR = "image-to-scene/indoor"
SETTING_OUTDOOR = "image-to-scene/outdoor"
SETTINGS = (SETTING_T2S, SETTING_INDOOR, SETTING_OUTDOOR)
I2S_SETTINGS = (SETTING_INDOOR, SETTING_OUTDOOR)

# The case schedule is data (benchmark/public-*-cases.txt), never a constant.
