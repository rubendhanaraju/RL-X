# Avoiding 2D

MuJoCo-free 2D point-mass version of the D3IL avoiding task. Actions are
D3IL-style Cartesian target deltas: each step adds the action to the previous
commanded target, then the point is moved toward that target over
`environment.n_substeps`.

Run with:

```bash
--environment.name=custom_jax.avoiding_2d
```
