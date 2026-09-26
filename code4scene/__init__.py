"""Code4Scene: verifiers and the scoring protocol of the Code4Scene benchmark.

Paper: "Code4Scene: Benchmarking Coding Agents for Constructing and Editing
3D Scenes". Coding agents build (text-to-scene) or repair (image-to-scene)
Unreal Engine 5.8 scenes; this package scores the saved scene.

Layout
------
``code4scene.evaluation``
    The verifier layer (Candidate Integrity, Physical Safety, the Semantic
    verifier for text-to-scene, and the ground-truth repair verifier).
``code4scene.protocol``
    The paper's exact scoring protocol: case scores, zero rules and the
    model score. Use this to reproduce the published numbers.
``code4scene.bundle``
    The ``code4scene.bundle.v1`` evidence-bundle format for offline scoring.
``code4scene.cli``
    The ``code4scene`` command line (``score``, ``rescore``, ``aggregate``).
"""

__version__ = "1.0.0"
PAPER_TITLE = (
    "Code4Scene: Benchmarking Coding Agents for Constructing and Editing 3D Scenes"
)
