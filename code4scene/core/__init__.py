"""Plumbing every layer needs and none of them owns.

Talking to a live editor, reading configuration, keeping a counter durable
across processes, stamping a record with what produced it, and reading the
scene graph out of an editor. If something here grows a policy rather than a
mechanism, it belongs in the layer that has the policy.
"""
