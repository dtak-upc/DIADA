Selection-only run of Experiment 4. The classification rows for mice_protein,
students, pendigits, dry_beans, default_payment, covertype and bank come from an
earlier pass that ranked the in-memory tables (before the runner was changed to
read every variant back from its CSV); they already matched the published
selected_features exactly and were kept rather than recomputed. Everything else
was computed with the current runner.
