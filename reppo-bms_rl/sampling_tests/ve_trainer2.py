from functools import partial

import hydra
import jax, wandb
import jax.numpy as jnp
import optax
from omegaconf import DictConfig, OmegaConf
from flax import nnx, struct

from src.jaxrl.reppo_ve_seperate_entropy_bound import ReppoConfig
from sampling_tests.cond_problems import get_problem
from sampling_tests.control_net import ControlNetwork
from sampling_tests.ve_sampler import VE, DIMEActor, adjoint_matching, ve_elbo, adjoint_matching_entr


class SACTrainState(struct.PyTreeNode):
    actor: nnx.TrainState
    actor_target: nnx.TrainState


def make_train_step(loss,  *, mini_batch_size: int, num_mini_batches: int, num_epochs_actor: int):
    total = mini_batch_size * num_mini_batches
    # -------------------------
    # JITted training step
    # -------------------------
    @partial(jax.jit)
    def train_step(
        train_state: SACTrainState,
        key: jax.Array,
        *,
        c_range: jax.Array
    ):
        """
        One outer training 'step' that runs:
          - sample full batch c
          - for each epoch:
              - shuffle once
              - scan over minibatches:
                  compute loss+grads
                  apply gradients
          - hard update target
        Returns: (new_state, mean_loss, key)
        """


        key, ckey, act_key = jax.random.split(key, 3)
        c = jax.random.uniform(
            ckey,
            shape=(total, 1),
            minval=c_range[0],
            maxval=c_range[1],
        )


        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        (action, prior_action, log_weights, log_path_weight_deterministic, log_path_weight_stochastic,
         log_p_T_ref) = actor_model.sde_sample(act_key, c, stop_grad=True)

        # jax.debug.print(
        #     "old_entropy={}",
        #     log_weights.sum(axis=1).mean()
        # )

        def run_one_epoch(carry, epoch_key):
            ts = carry

            # shuffle + reshape once per epoch (fast)
            perm = jax.random.permutation(epoch_key, total)
            c_shuf = c[perm].reshape((num_mini_batches, mini_batch_size, 1))
            a_shuf = action[perm].reshape((num_mini_batches, mini_batch_size, 1))
            prior_a_shuf = prior_action[perm].reshape((num_mini_batches, mini_batch_size, 1))
            log_weights_shuf = log_weights[perm].reshape((num_mini_batches, mini_batch_size, 1))
            log_path_weight_det_shuf = log_path_weight_deterministic[perm].reshape((num_mini_batches, mini_batch_size, 1))
            log_path_weight_sto_shuf = log_path_weight_stochastic[perm].reshape((num_mini_batches, mini_batch_size, 1))
            log_p_T_ref_shuf = log_p_T_ref[perm].reshape((num_mini_batches, mini_batch_size, 1))

            xs = (
                c_shuf,
                a_shuf,
                prior_a_shuf,
                log_weights_shuf,
                log_path_weight_det_shuf,
                log_path_weight_sto_shuf,
                log_p_T_ref_shuf,
            )

            # Better: carry a key through the minibatch scan
            def minibatch_scan_body(carry2, mb):
                ts_inner, k_inner = carry2
                k_inner, k_loss = jax.random.split(k_inner)
                (c_mb, a_mb, prior_a_mb,
                 log_weight_mb, log_path_weight_det_mb, log_path_weight_sto_mb, log_p_T_ref_mb) = mb

                # loss signature in your code: loss(key, actor_params, c_minibatch, train_state)
                (loss_val, aux), grads = loss(k_loss, ts_inner.actor.params, c_mb, ts_inner,
                                              a_mb, prior_a_mb, log_weight_mb, log_path_weight_det_mb,
                                              log_path_weight_sto_mb, log_p_T_ref_mb)
                new_actor_state = ts_inner.actor.apply_gradients(grads)
                ts_inner = ts_inner.replace(actor=new_actor_state)
                return (ts_inner, k_inner), (loss_val, aux)

            # epoch RNG stream
            epoch_key, mb_key = jax.random.split(epoch_key)
            (ts, _), (loss_vals, aux_vals) = jax.lax.scan(minibatch_scan_body, (ts, mb_key), xs)
            return ts, (aux_vals, jnp.mean(loss_vals))

        # Make epoch keys
        key, ekey = jax.random.split(key)
        epoch_keys = jax.random.split(ekey, num_epochs_actor)

        train_state, (metrics, epoch_losses) = jax.lax.scan(run_one_epoch, train_state, epoch_keys)

        # Hard update target
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(params=train_state.actor.params)
        )

        key, entrkey = jax.random.split(key)
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        entr = actor_model.lb_entropy(entrkey, c, stop_grad=True)
        #
        # jax.debug.print(
        #     "entropy_after_opt={}",
        #     entr
        # )
        metrics.update({'weights_entropy': entr})
        return train_state, jnp.mean(epoch_losses), metrics, key
    return train_step


# -------------------------
# JITted eval sampler (returns arrays to Python)
# -------------------------
@partial(jax.jit, static_argnames=("n_eval",))
def eval_sample(train_state: SACTrainState, key: jax.Array, c_range: jax.Array, n_eval: int):
    key, ckey, skey = jax.random.split(key, 3)
    c = jax.random.uniform(
        ckey,
        shape=(n_eval, 1),
        minval=c_range[0],
        maxval=c_range[1],
    )

    # Merge inside jit (fine); returns pure arrays.
    actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
    x, *_ = actor_model.sde_sample(skey, c, stop_grad=True)

    return c, x, key


@hydra.main(version_base=None, config_path="../config", config_name="reppo_ve_sampling_test")
def run_cond_tests(cfg: DictConfig) -> None:
    print(jax.default_backend())

    key, key_gen = jax.random.split(jax.random.PRNGKey(cfg.seed))
    # cfg = hydra.utils.instantiate(cfg)

    # target = get_problem('Const')
    target = get_problem('Sinusoidal')
    # Initialize actor
    key, diff_key = jax.random.split(key, 2)
    key, model_key = jax.random.split(key, 2)

    action_dim = 1
    observation_dim = 1

    log_wandb = cfg.wandb.mode == 'online'
    if log_wandb:
        wandb.init(
            mode=cfg.wandb.mode,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            tags=[
                cfg.name,
                cfg.env.name,
                cfg.env.type,
                *cfg.tags,
            ],
            config=OmegaConf.to_container(cfg),
            name=f"{cfg.name}-{cfg.env.name.lower()}"
        )


    cfg = ReppoConfig(**cfg.hyperparameters)

    noise_schedule = hydra.utils.call(cfg.diffusion.noise_schedule)
    sde_integrator = hydra.utils.get_method(cfg.diffusion.sde_integrator)
    sde_integrator_with_kl = hydra.utils.get_method(cfg.diffusion.sde_integrator_with_kl)
    ode_integrator = hydra.utils.get_method(cfg.diffusion.ode_integrator)
    logratio = hydra.utils.get_method(cfg.diffusion.logratio)

    forward_model = ControlNetwork(
        action_dim=action_dim,
        observation_dim=observation_dim,
        num_layers=cfg.diffusion.score_model.num_layers,
        num_hid = cfg.diffusion.score_model.num_hid,
        num_time_hid=cfg.diffusion.score_model.num_time_hid,
        num_time_out=cfg.diffusion.score_model.num_time_out,
        outer_clip=cfg.diffusion.score_model.outer_clip,
        inner_clip=cfg.diffusion.score_model.inner_clip,
        weight_init=cfg.diffusion.score_model.weight_init,
        bias_init=cfg.diffusion.score_model.bias_init,
        layer_norm=cfg.diffusion.score_model.layer_norm,
        layer_norm_type=cfg.diffusion.score_model.layer_norm_type,
        rngs=nnx.Rngs(model_key)
    )


    diffusion_model = VE(
        action_dim=1,
        observation_dim=1,
        fwd_model=forward_model,
        diff_steps=cfg.diffusion.diff_steps,
        scheduler=noise_schedule,
        rngs=nnx.Rngs(model_key),

    )

    actor_networks = DIMEActor(
        action_dim=action_dim,
        observation_dim=observation_dim,
        diffusion_model=diffusion_model,
        logratio=logratio,
        kl_start=cfg.kl_start,
        ent_start=cfg.ent_start,
        sde_integrator=sde_integrator,
        sde_integrator_with_kl=sde_integrator_with_kl,
        ode_integrator=ode_integrator,
        kl_bound=cfg.kl_bound,
        entropy_constraint=cfg.entropy_constraint
    )

    actor_target_networks = DIMEActor(
        action_dim=action_dim,
        observation_dim=observation_dim,
        diffusion_model=diffusion_model,
        logratio=logratio,
        kl_start=cfg.kl_start,
        ent_start=cfg.ent_start,
        sde_integrator=sde_integrator,
        sde_integrator_with_kl=sde_integrator_with_kl,
        ode_integrator=ode_integrator,
        kl_bound=cfg.kl_bound,
    )

    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(3.0e-4)
    )

    actor_trainstate = nnx.TrainState.create(
        graphdef=nnx.graphdef(actor_networks),
        params=nnx.state(actor_networks),
        tx=actor_optimizer,
    )
    actor_target_trainstate = nnx.TrainState.create(
        graphdef=nnx.graphdef(actor_target_networks),
        params=nnx.state(actor_target_networks),
        tx=optax.set_to_zero(),
    )

    train_state= SACTrainState(
        actor=actor_trainstate,
        actor_target=actor_target_trainstate
    )


    if cfg.entropy_constraint:
        loss_short = partial(adjoint_matching_entr, target_log_prob=target.log_prob, get_Q= target.get_Q,
                             batch_repetitions=cfg.batch_repetitions, update_kl_lagrangian=cfg.update_kl_lagrangian,
                             kl_bound=cfg.kl_bound, entr_bound=cfg.entr_bound)
    else:
        loss_short = partial(adjoint_matching, target_log_prob=target.log_prob, get_Q= target.get_Q,
                             batch_repetitions=cfg.batch_repetitions, update_kl_lagrangian=cfg.update_kl_lagrangian,
                             kl_bound=cfg.kl_bound)
    elbo = jax.jit(partial(ve_elbo, target_log_prob=target.log_prob))
    loss = jax.jit(jax.value_and_grad(loss_short, 1, has_aux=True))

    c_range = jnp.array([-3, 3])

    # -------------------------
    # Python loop: fast training, plotting at checkpoints
    # -------------------------
    N_ITERS = 5000
    N_EVALS = 10
    eval_freq = max(N_ITERS // N_EVALS, 1)

    mini_batch_size = 1000
    num_mini_batches = cfg.num_mini_batches

    train_step = make_train_step(loss, mini_batch_size = mini_batch_size,
                                       num_mini_batches = int(cfg.num_mini_batches),
                                       num_epochs_actor = int(cfg.num_epochs_actor))




    for step in range(N_ITERS):
        train_state, loss_value, metrics, key = train_step(train_state, key, c_range=c_range)

        # Optional: print (forces a sync when you device_get, so keep it infrequent)
        if step % 1 == 0:
            # # If you want ELBO too, keep it as a separate jitted function like eval_sample
            # # so training stays clean.
            # elbo_val, running_costs, terminal_costs = elbo(key, train_state.actor.params,  # type: ignore
            #                                               jax.random.uniform(key, (mini_batch_size * num_mini_batches, 1),
            #                                                                  minval=c_range[0], maxval=c_range[1]),
            #                                               train_state)

            if not cfg.entropy_constraint:
                print("")
                # print(f"step={step} loss={float(jax.device_get(loss_value))} ELBO={float(jax.device_get(-elbo_val))}")
                print(f"step={step} kl={float(jax.device_get(jnp.mean(jnp.mean(metrics['kl'], axis=1))))} kl_LM={jax.device_get(jnp.mean(jnp.mean(metrics["m_step_lagrangian"])))}")
                print(f"step={step} e_step_TRLM={float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_lagrangian'], axis=1))))}")
                print(f"step={step} entropy={float(jax.device_get(jnp.mean(jnp.mean(metrics['entropy'], axis=1))))}")
                print("")
            else:
                print("")
                # print(f"step={step} loss={float(jax.device_get(loss_value))} ELBO={float(jax.device_get(-elbo_val))}")
                print(f"step={step} kl={float(jax.device_get(jnp.mean(jnp.mean(metrics['kl'], axis=1))))} kl_LM={jax.device_get(jnp.mean(jnp.mean(metrics["m_step_lagrangian"])))}")
                print(f"step={step} e_step_TRLM={float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_lagrangian'], axis=1))))}")
                print(f"step={step} e_step_ENTRLM={float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_lagrangian_entr'], axis=1))))}")
                print(f"step={step} e_step_ENTRLM_LB={float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_lagrangian_lb_entr'], axis=1))))}")
                print(f"step={step} e_step_dual_val={float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_dual_val'], axis=1))))}")
                print(f"step={step} entropy={float(jax.device_get(jnp.mean(jnp.mean(metrics['entropy'], axis=1))))}")
                print(f"step={step} weights_entropy={float(jax.device_get(jnp.mean(metrics['weights_entropy'])))}")
                print(f"step={step} entropy_unsmoothed={float(jax.device_get(jnp.mean(jnp.mean(metrics['entropy_unsmoothed'], axis=1))))}")
                print(f"step={step} old_entropy={float(jax.device_get(jnp.mean(jnp.mean(metrics['old_entropy'], axis=1))))}")
                print(f"step={step} entropy_bound={float(jax.device_get(jnp.mean(jnp.mean(metrics['entr_bound'], axis=1))))}")
                print(f"step={step} SmoothedESS={float(jax.device_get(jnp.mean(jnp.mean(metrics['smoothed_ESS'], axis=1))))}")
                print(f"step={step} ESS={float(jax.device_get(jnp.mean(jnp.mean(metrics['ESS'], axis=1))))}")
                # print(f"step={step} logZ={float(jax.device_get(jnp.mean(jnp.mean(metrics['logZ'], axis=1))))}")
                print(f"step={step} reward_mean={float(jax.device_get(jnp.mean(jnp.mean(metrics['reward_mean'], axis=1))))}")
                print(f"step={step} lbfgs_grad={float(jax.device_get(jnp.mean(jnp.mean(metrics['lbfgs_grad'], axis=1))))}")
                print(f"step={step} lbfgs_num_updates={float(jax.device_get(jnp.mean(jnp.mean(metrics['lbfgs_num_updates'], axis=1))))}")
                print(f"step={step} lbfgs_error={float(jax.device_get(jnp.mean(jnp.mean(metrics['lbfgs_error'], axis=1))))}")
                print(f"step={step} log_weights={float(jax.device_get(jnp.mean(jnp.mean(metrics['log_weights'], axis=1))))}")
                wandb.log({"step": step,
                           'loss': float(jax.device_get(jnp.mean(jnp.mean(metrics['loss'], axis=1)))),
                           'entropy': float(jax.device_get(jnp.mean(jnp.mean(metrics['entropy'], axis=1)))),
                           'entropy_weights': float(jax.device_get(jnp.mean(metrics['weights_entropy']))),
                           'entropy_bound': float(jax.device_get(jnp.mean(jnp.mean(metrics['entr_bound'], axis=1)))),
                           'old_entropy': float(jax.device_get(jnp.mean(jnp.mean(metrics['old_entropy'], axis=1)))),
                           'e_step_ENTRLM': float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_lagrangian_entr'], axis=1)))),
                           'e_step_ENTRLM_LB': float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_lagrangian_lb_entr'], axis=1)))),
                           'e_step_TRLM': float(jax.device_get(jnp.mean(jnp.mean(metrics['e_step_lagrangian'], axis=1)))),
                           'm_step_TRLM': float(jax.device_get(jnp.mean(jnp.mean(metrics['m_step_lagrangian'], axis=1)))),
                           'm_step_lagrangian_loss': float(jax.device_get(jnp.mean(jnp.mean(metrics['m_step_lagrangian_loss'], axis=1)))),
                })
                print("")
        # Checkpoint eval + interactive plot OUTSIDE jit
        if (step % eval_freq == 0) or (step == N_ITERS - 1):
            c_eval, x_eval, key = eval_sample(train_state, key, c_range, n_eval=100)
            # Bring to host for plotting
            c_host, x_host = jax.device_get((c_eval, x_eval))
            target.visualize(c_host, x_host)
    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    run_cond_tests()
