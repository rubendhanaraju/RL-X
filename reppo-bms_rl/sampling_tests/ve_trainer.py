from functools import partial

import hydra
import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig
from flax import nnx, struct

from src.jaxrl.reppo_ve_seperate_entropy_bound import ReppoConfig
from sampling_tests.cond_problems import get_problem
from sampling_tests.control_net import ControlNetwork
from sampling_tests.ve_sampler import VE, DIMEActor, adjoint_matching, ve_elbo



class SACTrainState(struct.PyTreeNode):
    actor: nnx.TrainState
    actor_target: nnx.TrainState



@hydra.main(version_base=None, config_path="../config", config_name="reppo_ve_sampling_test")
def run_cond_tests(cfg: DictConfig) -> None:
    print(jax.default_backend())

    key, key_gen = jax.random.split(jax.random.PRNGKey(cfg.seed))
    # cfg = hydra.utils.instantiate(cfg)

    target = get_problem('Sinusoidal')
    # target = get_problem('Sinusoidal')
    # Initialize actor
    key, diff_key = jax.random.split(key, 2)
    key, model_key = jax.random.split(key, 2)

    action_dim = 1
    observation_dim = 1
    num_layers = 3
    num_hid = 256
    num_time_hid = 32
    num_time_out = 16
    outer_clip = 1e4
    inner_clip = 1e2
    weight_init = 1e-8
    bias_init = 0.
    layer_norm = False
    layer_norm_type = "LayerNorm"

    diff_steps = 16
    batch_repetitions = 2
    update_kl_lagrangian = True
    kl_bound = 0.1

    cfg = ReppoConfig(**cfg.hyperparameters)

    noise_schedule = hydra.utils.call(cfg.diffusion.noise_schedule)
    sde_integrator = hydra.utils.get_method(cfg.diffusion.sde_integrator)
    sde_integrator_with_kl = hydra.utils.get_method(cfg.diffusion.sde_integrator_with_kl)
    ode_integrator = hydra.utils.get_method(cfg.diffusion.ode_integrator)
    logratio = hydra.utils.get_method(cfg.diffusion.logratio)

    forward_model = ControlNetwork(
        action_dim=action_dim,
        observation_dim=observation_dim,
        num_layers=num_layers,
        num_hid = num_hid,
        num_time_hid=num_time_hid,
        num_time_out=num_time_out,
        outer_clip=outer_clip,
        inner_clip=inner_clip,
        weight_init=weight_init,
        bias_init=bias_init,
        layer_norm=layer_norm,
        layer_norm_type=layer_norm_type,
        rngs=nnx.Rngs(model_key)
    )


    diffusion_model = VE(
        action_dim=1,
        observation_dim=1,
        fwd_model=forward_model,
        diff_steps=diff_steps,
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

    if cfg.max_grad_norm is not None:
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

    loss_short = partial(adjoint_matching, target_log_prob=target.log_prob, batch_repetitions=batch_repetitions,
                         update_kl_lagrangian=update_kl_lagrangian, kl_bound=kl_bound)
    elbo = jax.jit(partial(ve_elbo, target_log_prob=target.log_prob))
    loss = jax.jit(jax.value_and_grad(loss_short, 1, has_aux=True))

    c_range = jnp.array([-3, 3])

    N_ITERS = 10000
    N_EVALS = 20
    eval_freq = max(N_ITERS // N_EVALS, 1)


    for step in range(N_ITERS):
        # Evaluation step
        if (step % eval_freq == 0) or (step == N_ITERS - 1):
            # Update step
            key, key_gen = jax.random.split(key_gen)
            c = jax.random.uniform(key, shape=(100, 1), minval=c_range[0], maxval=c_range[1])
            key, key_gen = jax.random.split(key_gen)
            actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
            x, *_ = actor_model.sde_sample(key, c, stop_grad=True)
            key, key_gen = jax.random.split(key_gen)
            target.visualize(c, x)

        # Update step
        key, key_gen = jax.random.split(key_gen)
        # Sample from the conditional distribution
        mini_batch_size = 100
        c = jax.random.uniform(key, shape=(mini_batch_size*cfg.num_mini_batches , 1), minval=c_range[0], maxval=c_range[1])

        @jax.jit
        def update_sampler(train_state, key_gen):
            key, key_gen = jax.random.split(key_gen)
            def mini_batch_update(carry, indices):
                c_train_state = carry
                # Sample data at indices from the batch
                c_minibatch = jax.tree.map(lambda x: jnp.take(x, indices, axis=0), c)

                loss_value, grads = loss(key, c_train_state.actor.params, c_minibatch, c_train_state)
                actor_train_state = c_train_state.actor.apply_gradients(grads)
                c_train_state = c_train_state.replace(actor=actor_train_state)
                return c_train_state, loss_value

            shuffle_key, key_gen = jax.random.split(key_gen)
            indices = jax.random.permutation(shuffle_key, c.shape[0])
            minibatch_idx = jax.tree.map(lambda x: x.reshape((cfg.num_mini_batches, mini_batch_size)), indices)
            train_state, info = jax.lax.scan(mini_batch_update, train_state, xs=minibatch_idx)
            return train_state, info

        key, key_gen = jax.random.split(key_gen)
        train_state, info = jax.lax.scan(f=update_sampler,init=train_state, xs=jax.random.split(key, cfg.num_epochs_actor))
        loss_value = jnp.mean(info[0])
        metrics = info[1]

        if (step % 10 == 0):
            key, key_gen = jax.random.split(key_gen)
            elbo_val, running_costs, terminal_costs = elbo(key, train_state.actor.params, c, train_state)
            print(f'Loss {loss_value}, ELBO: {-elbo_val}, RC {running_costs}, TC {terminal_costs}')
            print("")

        train_state = train_state.replace(actor_target=train_state.actor_target.replace(params=train_state.actor.params))


if __name__ == "__main__":
    run_cond_tests()