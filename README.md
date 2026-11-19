# DFP_Ui

A User interface for the [Dubins Fleet Planner](https://github.com/enacuavlab/DubinsFleetPlanner)

## Batch plotting

Use `src/PlotPlanBatch.sh` to process every CSV or JSON trajectory file in a
folder:

```bash
src/PlotPlanBatch.sh [-R] [-s|--save 'results/%.png'] folder [plot-options...]
```

`-R` searches the folder recursively. `--save` requires a percent placeholder;
`%` is replaced with each input filename without its extension. Figures are
saved and never displayed. Without `--save`, temporary files are used and
removed after the batch completes. The script writes a CSV summary to stdout
and forwards Python output to stderr.
