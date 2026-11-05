"""The local web page for the agent pipeline (Part B).

`server.py` serves `static/` on 127.0.0.1 and streams the nine stages of one
case as server-sent events; `app.py` holds the testable logic; `serialize.py`
turns stage events into JSON; `outcome.py` looks up the real outcome for
display only. See PlanDoc/agentic_with_fe.md §14 and frontend/README.md.
"""
