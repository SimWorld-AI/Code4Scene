from code4scene.evaluation.requirement_graph.actor_inventory import (
    IdentitySource,
    build_actor_descriptor,
    descriptor_from_mapping,
)


def test_non_semantic_generated_label_is_recorded_and_skipped():
    descriptor = build_actor_descriptor(
        live_actor_id="StaticMeshActor_451",
        actor_label="C1_c",
        asset_path=(
            "/Game/Village/Meshes/SM_Loghouse01_SidingBoards_CapsCorners01."
            "SM_Loghouse01_SidingBoards_CapsCorners01"
        ),
    )

    assert all(
        not (
            term.source is IdentitySource.ACTOR_LABEL
            and term.raw_value == "C1_c"
        )
        for term in descriptor.identity_terms
    )
    assert any(
        term.source is IdentitySource.ASSET_PATH
        for term in descriptor.identity_terms
    )
    assert descriptor.identity_term_diagnostics == (
        "skipped actor_label identity term 'C1_c': "
        "identity term must contain semantic text",
    )


def test_explicit_non_semantic_term_is_recorded_without_aborting_descriptor():
    descriptor = build_actor_descriptor(
        live_actor_id="actor-1",
        asset_path="/Game/Props/SM_Chair.SM_Chair",
        identity_terms=("Actor",),
    )

    assert "chair" in descriptor.identity_values
    assert descriptor.identity_term_diagnostics == (
        "skipped structured_tag identity term 'Actor': "
        "identity term must contain semantic text",
    )


def test_identity_term_diagnostics_round_trip_with_trusted_serialized_terms():
    original = build_actor_descriptor(
        live_actor_id="actor-1",
        actor_label="C1_c",
        asset_path="/Game/Props/SM_Chair.SM_Chair",
    )

    restored = descriptor_from_mapping(
        original.to_dict(),
        trust_serialized_terms=True,
    )

    assert restored.to_dict() == original.to_dict()
