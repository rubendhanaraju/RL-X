# D3IL MJX Environments

This folder contains MJX-native, fully JIT-able RLX versions of the planar D3IL manipulation tasks:

- `custom_mujoco.d3il.avoiding.mjx`
- `custom_mujoco.d3il.pushing.mjx`
- `custom_mujoco.d3il.aligning.mjx`
- `custom_mujoco.d3il.sorting.mjx`
- `custom_mujoco.d3il.stacking.mjx`
- `custom_mujoco.d3il.inserting.mjx`

The original D3IL Gym tasks use a Python Cartesian impedance controller around a Panda robot. These RLX environments keep the D3IL task layouts, planar action semantics, observations, rewards, success checks, randomized contexts, and mode bookkeeping, but run the transition through MJX contact dynamics rather than the previous kinematic object updates. Rendering uses copied D3IL MuJoCo XML assets under `assets/` for the Panda, table, task objects, targets, and obstacles, so the envs no longer depend on the original `d3il/` checkout. MJX does not support all MuJoCo collision pairs used by the original XML, so visual Panda meshes and decorative cage geometry are collision-disabled; task-relevant contacts use the table, task objects, rod/finger collision geoms, obstacles, maze walls, sorting platform, and sorting target walls. The default renderer uses the visible Panda XML variants; set `environment.render_visible_robot=False` to use D3IL's translucent `*_invisible.xml` variants.

Task-specific behavior lives in each env's `mjx/environment.py`. `common_mjx` contains only shared RLX/MJX runtime plumbing, XML assembly helpers, spaces, and state definitions.

Use them with a JAX full-JIT algorithm, for example:

```bash
--algorithm.name=ppo.flax_full_jit --environment.name=custom_mujoco.d3il.avoiding.mjx
```

To sanity-check rendering with a random policy:

```bash
python rl_x/environments/custom_mujoco/d3il/visualize_random_policy.py --environment custom_mujoco.d3il.pushing.mjx
```
