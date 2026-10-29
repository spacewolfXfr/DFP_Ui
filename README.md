# DFP_Ui

A User interface for the [Dubins Fleet Planner](https://github.com/enacuavlab/DubinsFleetPlanner)

## Scripts

The executable scripts in `src/` are thin entry points and demonstrations. Run them from the repository root, for example:

```bash
python src/PlotPlan.py --help
```

### Plan and traffic tools

- `PlotPlan.py` displays a Dubins flight plan from a JSON or CSV trajectory file. It can create an animation or a static snapshot, and supports dynamic and geometric obstacles, custom axis limits, sampling, colormaps, and saving the output.

    ```bash
    python src/PlotPlan.py PLAN.json --static 8 --geo-obstacles geometric_obstacles.json
    ```

- `TrafficFetcher.py` downloads traffic around an airport through the OpenSky API and writes it to a Parquet file. The required arguments are an airport ICAO code and a start timestamp; use `--end` or `--duration` to define the time window.

    ```bash
    python src/TrafficFetcher.py EGLL 2026-03-16T17:14:00 --duration 60 --output traffic.parquet
    ```

- `PlotTraffic.py` plots or animates a traffic dataset in Parquet format for one or more airport ICAO codes. It can compare a second dataset and save the generated figures or animation.

    ```bash
    python src/PlotTraffic.py traffic.parquet EGLL --animate
    ```

- `TrafficSimulator.py` runs the arrival/departure traffic simulation and invokes an external Dubins Fleet Planner solver. It supports traffic periods, separation settings, lookahead and rescheduling controls, optional obstacle boundaries, data export, profiling, and an interactive UI.

    ```bash
    python src/TrafficSimulator.py /path/to/DubinsFleetPlanner --data data/traffic.parquet -ui
    ```

- `PlotSimSummary.py` visualizes the simulation summary stored in the pickle file produced by `TrafficSimulator.py`.

    ```bash
    python src/PlotSimSummary.py simulation_summary.pkl
    ```

### Demonstrations

- `demo_obstacles.py` runs a self-contained Matplotlib example showing a planned SLS path, a straight obstacle, and the path drift caused by a constant wind vector. It opens a plot directly and does not take command-line arguments.

    ```bash
    python src/demo_obstacles.py
    ```

- `minifig.py` is an exploratory plotting script containing hard-coded path sections. It compares two trajectories, reports their sampled minimum distance, and displays the result; it is intended for development experiments rather than general-purpose use.

    ```bash
    python src/minifig.py
    ```

The implementation modules under `src/DubinsFleetPlanner/` are not listed here; the scripts above dispatch to those modules where applicable. Use `--help` on the command-line tools for the complete option list.
