# RoboCup Pose Studio

Standalone pose and keyframe editor for the RoboCup Booster T1 robot.

It opens a MuJoCo viewer, lets you manipulate:
- root position
- root orientation
- all 23 actuator joints

It also supports:
- pose export/import as JSON
- keyframe timelines
- animation playback with interpolation
- MJCF `<keyframe>` block export
- undo/redo with `Ctrl+Z` / `Ctrl+Y`

## Setup

Create the Conda environment from the project root:

```bash
cd robocup_pose_studio
conda env create -f environment.yml
conda activate robocup-pose-studio
```

## Run

```bash
conda activate robocup-pose-studio
robocup-pose-studio
```

By default it uses the Booster T1 XML already present in this repository.

You can also point it at another XML:

```bash
robocup-pose-studio --xml /path/to/plane.xml
```

Or load a saved pose / animation on startup:

```bash
robocup-pose-studio --load my_animation.json
```

## Notes

- This project depends on a working GUI session. On headless machines, `glfw` will complain about missing display access.
- The Booster T1 robot XML and mesh assets are bundled with this package, so it can run independently of the RL-X repo once installed.
- If you split this folder into its own Git repository, the bundled assets and `environment.yml` are enough to bootstrap it without the rest of RL-X.
