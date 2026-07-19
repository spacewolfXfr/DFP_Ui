# AGENTS.md — Chat agent instructions for DFP_Ui

Purpose
- Help AI coding agents become productive quickly in this repository.
- Keep guidance minimal, link-heavy, and actionable.

Quick context
- Project: DFP_Ui — a User Interface for the Dubins Fleet Planner. See [README.md](README.md).
- Primary code: `src/` (core modules, plotting, UI, traffic tools).
- Ignore the content of `data/` unless explicitly specified

How to get started (agent checklist)
- Read the README: [README.md](README.md).
- Inspect these entry points and helpful files:
  - [src/plotting.py](src/plotting.py): plotting utilities used by the UI.
  - [src/UI](src/UI): UI helpers and slider widgets.
  - [src/Aircraft.py](src/Aircraft.py): aircraft model and state.
  - [src/Formation.py](src/Formation.py): formation-related logic.
  - [src/traffic](src/traffic): traffic generator, simulator, and player scripts.
- Run quick, low-risk commands when needed (from repo root):

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt  # if present; otherwise install needed libs (matplotlib, numpy, scipy, pandas)
  python src/traffic/traffic_player.py
  python src/traffic/traffic_pb_simulator.py
  ```

Conventions and expectations for agents
- Link, don't embed: prefer linking to existing docs/files rather than copying large content.
- Preserve style: follow existing Python style and repository layout; do not reorganize files unless asked.
- Tests/builds: there are no explicit test harnesses; if you add tests, update this file with instructions.
- Make minimal, focused changes. Explain intent in PR descriptions.

When creating new instructions or skills
- Add small, targeted docs under `.github/` or add a new skill file only when it helps automation (e.g., CI hooks, common refactors).

If unsure
- Ask a clarifying question before making broad changes (project goals, preferred Python version, CI expectations).

Contact
- No maintainers listed in repo; open a PR with changes and request review.
