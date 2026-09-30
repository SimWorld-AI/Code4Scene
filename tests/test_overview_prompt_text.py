"""The overview judge reads only the scene description, not the agent's build instructions."""

from __future__ import annotations

from code4scene.evaluation.offline_overview_prompt_alignment import _prompt_for_overview

SCENE = "Build me a quiet harbour town with red roofs and a stone quay."


def test_the_asset_palette_section_is_removed():
    assert _prompt_for_overview(SCENE + "\n\n=== ASSET PALETTE ===\n- crate | ...") == SCENE


def test_an_indented_ground_section_is_removed():
    prompt = SCENE + "\n\n    === GROUND ===\n    Build the FLOOR FIRST: call ground_pass.\n\n=== ASSET PALETTE ===\n- tile"
    assert _prompt_for_overview(prompt) == SCENE


def test_a_prompt_without_agent_sections_is_unchanged():
    assert _prompt_for_overview("  " + SCENE + "\n") == SCENE
