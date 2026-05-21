python src/jaxrl/reppo_ve_seperate.py -m seed=0 \
hyperparameters.diffusion.loss=rkl,am \
env.name=CheetahRun,FishSwim,HopperHop,CartpoleSwingup,WalkerRun \
tags='["bm-1"]' \
+launcher=slurm \
