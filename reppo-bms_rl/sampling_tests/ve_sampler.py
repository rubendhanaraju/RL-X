import math
from functools import partial
from typing import NamedTuple
# from jax._src.scipy.special import logsumexp
from jax.scipy.special import logsumexp

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from src.jaxrl.lagrangian_utils import bracket_search_minimizer, tr_dual_fn, compute_tr_lm, tr_ent_dual_fn, \
    compute_tr_ent_lm
from src.jaxrl.utils import compute_reverse_ess


class VE(nnx.Module):
    def __init__(
            self,
            action_dim: int,
            observation_dim: int,
            fwd_model: nnx.Module = None,
            diff_steps: int = 8,
            scheduler: NamedTuple = None,
            *,
            rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diff_steps = diff_steps
        self.fwd_model = fwd_model
        # self.dt_schedule = dt_schedule
        self.noise_scheduler = scheduler

        dt = 1. / diff_steps
        self.dt = dt

        prior = distrax.MultivariateNormalDiag(
            jnp.zeros(action_dim), jnp.ones(action_dim) * self.noise_scheduler.sigma_T_0()
        )

        # jax.debug.print(f'Prior STD for variance exploding SDE is {self.noise_scheduler.sigma_T_0()}')
        @jax.jit
        def _prior_log_prob_fn(x):  # todo
            log_probs = prior.log_prob(x)
            return log_probs

        @partial(jax.jit, static_argnames=["n_samples"])
        def _prior_sampler_fn(key, n_samples):
            samples = prior.sample(seed=key, sample_shape=(n_samples,))
            return samples

        self.prior_log_prob = _prior_log_prob_fn
        self.prior_sampler = _prior_sampler_fn

    def forward_model(
            self, step: jax.Array, x: jax.Array, obs: jax.Array, aux: jax.Array = None
    ) -> jax.Array:
        """Forward model function."""
        if self.fwd_model is not None:
            return self.fwd_model(x, obs, step)
        else:
            return jnp.zeros_like(x)


class DIMEActor(nnx.Module):
    def __init__(
            self,
            action_dim: int,
            observation_dim: int,
            diffusion_model: nnx.Module,
            sde_integrator: callable,
            sde_integrator_with_kl: callable,
            ode_integrator: callable,
            logratio: callable,
            kl_start: float = 0.1,
            ent_start: float = 0.1,
            kl_bound: float = 0.1,
            entropy_constraint: bool = False,
    ):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.diffusion_model = diffusion_model
        self.sde_integrator = sde_integrator
        self.sde_integrator_with_kl = sde_integrator_with_kl
        self.ode_integrator = ode_integrator
        self.logratio = logratio

        # Parameters
        self.log_lagrangian = nnx.Param(jnp.ones(1) * math.log(kl_start))
        self.log_temperature = nnx.Param(jnp.ones(1) * math.log(ent_start))

        self.entropy_temperature = jnp.ones(1)

        if entropy_constraint:
            # Note: the entropy bound is calculated dynamically because of linear decay in entropy constrinat ...
            dual = partial(tr_ent_dual_fn, kl_bound=kl_bound)
            self.optimize_lm = jax.jit(partial(compute_tr_ent_lm, dual_fn=dual))
        else:
            dual = partial(tr_dual_fn, tr_bound=kl_bound)
            self.optimize_lm = jax.jit(partial(compute_tr_lm, dual_fn=dual))

    def ode_sample(
            self,
            key,
            obs: jax.Array,
            stop_grad: bool = False,
            ode_coef: float = 1.0,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        keys = jax.random.split(key, num=obs.shape[0])

        def _ode_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, key_aux)

            integrate = self.ode_integrator(obs, self.diffusion_model, stop_grad, ode_coef)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, _ = aux

            final_x = distrax.Tanh().forward(final_x)
            return final_x

        x_0 = jax.vmap(_ode_sample, in_axes=(0, 0))(keys, obs)
        return x_0

    def sde_sample(
            self,
            key,
            obs: jax.Array,
            stop_grad: bool = False,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        keys = jax.random.split(key, num=obs.shape[0])

        def _sde_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            prior_log_prob = self.diffusion_model.prior_log_prob(init_x)

            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), jnp.zeros(1), key_aux)

            integrate = self.sde_integrator(obs, self.diffusion_model, stop_grad=stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, log_path_weight_deterministic, log_path_weight_stochastic, _ = aux

            prior_log_weight = prior_log_prob.reshape(log_path_weight_deterministic.shape)
            # log (dQ/dP^u)|T - log p_T_ref(X_T)  = -∫ ½||u(X_t,t)||^2 dt - ∫ u(X_t,t) dB_t - log p_T_ref(X_T)
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic - prior_log_weight

            # running_cost = -(log_path_weight_deterministic + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
            # terminal_costs = self.diffusion_model.ref_log_prob(final_x)
            # stochastic_costs = log_path_weight_stochastic

            # return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape)
            return final_x, init_x, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, prior_log_prob

        rnd_result = jax.vmap(_sde_sample, in_axes=(0, 0))(keys, obs)
        # x_0, running_costs, stochastic_costs, terminal_costs = rnd_result
        x_0, init_x, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, log_p_T_ref = rnd_result
        return x_0, init_x, log_weight, log_path_weight_deterministic, log_path_weight_stochastic, log_p_T_ref


    def lb_entropy(self,
                   key,
                   obs,
                   stop_grad):

        _, _, log_weight, _, _, _ = self.sde_sample(key, obs, stop_grad)
        return log_weight.mean() # the log_weights are already negated...

    def sde_sample_and_kl(
            self,
            key,
            obs: jax.Array,
            target_actor: nnx.Module,
            stop_grad: bool = False,
    ) -> jax.Array:
        """Sample actions from the SDE diffusion model."""
        target_diffusion_model = target_actor.diffusion_model if hasattr(target_actor,
                                                                         'diffusion_model') else target_actor

        keys = jax.random.split(key, num=obs.shape[0])

        def _sde_sample(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), key_aux)

            integrate = self.sde_integrator_with_kl(obs, target_diffusion_model, self.diffusion_model,
                                                    stop_grad=stop_grad)
            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, log_path_weight_deterministic, log_path_weight_stochastic, kl_weight, _ = aux

            log_p_T_ref = self.diffusion_model.ref_log_prob(final_x).reshape(log_path_weight_deterministic.shape)

            # log (dQ/dP^u)|T - log p_T_ref(X_T)  = -∫ ½||u(X_t,t)||^2 dt - ∫ u(X_t,t) dB_t - log p_T_ref(X_T)
            log_weight = log_path_weight_deterministic + log_path_weight_stochastic - log_p_T_ref + distrax.Tanh().forward_log_det_jacobian(
                final_x).sum()

            # running_cost = -(log_path_weight_deterministic + distrax.Tanh().forward_log_det_jacobian(final_x).sum())
            # terminal_costs = self.diffusion_model.ref_log_prob(final_x)
            # stochastic_costs = log_path_weight_stochastic
            final_x = distrax.Tanh().forward(final_x)

            # return final_x, running_cost, stochastic_costs, terminal_costs.reshape(running_cost.shape)
            return final_x, log_weight, kl_weight

        rnd_result = jax.vmap(_sde_sample, in_axes=(0, 0))(keys, obs)
        # x_0, running_costs, stochastic_costs, terminal_costs = rnd_result
        x_0, log_weight, kl_weight = rnd_result
        return (x_0, log_weight, kl_weight)

    def kl_div_dime(self, key, obs: jax.Array, target_actor: nnx.Module, stop_grad: bool = False) -> jax.Array:
        """
        Compute KL divergence using the SINGLE PATH integrator.

        Args:
            key: Random key
            obs: Observations
            target_actor: Target DIMEActor (we extract its diffusion_model)
            stop_grad: Whether to stop gradients

        Note: Cannot JIT this function because it takes module arguments.
        The outer JIT context (from actor_loss) will handle compilation.
        """
        # Extract diffusion_model from target_actor
        target_diffusion_model = target_actor.diffusion_model if hasattr(target_actor,
                                                                         'diffusion_model') else target_actor
        keys = jax.random.split(key, num=obs.shape[0])

        def _kl_div_dime(key, obs):
            key, key_init, key_aux = jax.random.split(key, 3)
            init_x = self.diffusion_model.prior_sampler(key_init, 1)
            init_x = jnp.squeeze(init_x, 0)
            if stop_grad:
                init_x = jax.lax.stop_gradient(init_x)
            aux = (init_x, jnp.zeros(1), key_aux)

            integrate = self.logratio(
                self.diffusion_model,
                target_diffusion_model,
                obs,
                stop_grad=stop_grad
            )

            aux, _ = jax.lax.scan(integrate, aux, jnp.arange(0, self.diffusion_model.diff_steps))
            final_x, log_ratio, _ = aux

            return log_ratio

        log_ratios = jax.vmap(_kl_div_dime, in_axes=(0, 0))(keys, obs)
        return log_ratios

    def set_temperature(self, temperature):
        self.entropy_temperature = temperature

    def temperature(self) -> jax.Array:
        return self.entropy_temperature

    # def temperature(self) -> jax.Array:
    #     return jnp.exp(self.log_temperature.value)

    def lagrangian(self) -> jax.Array:
        return jnp.exp(self.log_lagrangian.value)



def ve_elbo(key, params, contexts, train_state, target_log_prob):
    actor_model = nnx.merge(train_state.actor.graphdef, params)
    action, prior_action,log_weights, log_path_weight_deterministic, log_path_weight_stochastic, log_p_T_ref \
        = actor_model.sde_sample(key, contexts, stop_grad=True)
    elbo = log_path_weight_deterministic + log_p_T_ref + target_log_prob(action, contexts)
    return jnp.mean(elbo), log_path_weight_deterministic.mean(), (log_p_T_ref + target_log_prob(action, contexts) ).mean()



def adjoint_matching(key, params, contexts, train_state, action, prior_action, log_weights, log_path_weight_deterministic,
                     log_path_weight_stochastic, log_p_T_ref, target_log_prob, batch_repetitions, update_kl_lagrangian, kl_bound):
    actor_model = nnx.merge(train_state.actor.graphdef, params)
    actor_target_model = nnx.merge(
        train_state.actor_target.graphdef, train_state.actor_target.params
    )
    # Access the diffusion components
    diffusion = actor_model.diffusion_model
    old_diffusion = actor_target_model.diffusion_model
    scheduler = diffusion.noise_scheduler

    batch_size = contexts.shape[0] * batch_repetitions

    key, act_key = jax.random.split(key, 2)
    key, act_key, act_rndm = jax.random.split(key, 3)

    # action, prior_action, log_weights, log_path_weight_deterministic, log_path_weight_stochastic, log_p_T_ref = actor_model.sde_sample(
    #     act_key, contexts, stop_grad=True)

    a_0 = jnp.repeat(prior_action, batch_repetitions, axis=0)
    action = jnp.repeat(action, batch_repetitions, axis=0)
    contexts = jnp.repeat(contexts, batch_repetitions, axis=0)

    Q_value, Q_score = jax.vmap(jax.value_and_grad(lambda x, c: target_log_prob(x, c).squeeze()), in_axes=(0, 0))(action, contexts)

    Q_value = Q_value.reshape(Q_score.shape)
    log_weights = jnp.repeat(log_weights, batch_repetitions, axis=0)

    # 1. Randomly sample time t between [0,1]
    key_t, key_noise = jax.random.split(key, 2)  # Split the key passed to actor_loss
    t = jax.random.uniform(key_t, (batch_size, 1))

    # 2. Randomly sample noise from N(0, I)
    noise = jax.random.normal(key_noise, action.shape)

    # 3. Sample a_t (Forward Diffusion Process)
    # a_T is the clean action from the buffer (minibatch.action)

    # Retrieve coefficients from scheduler (handling broadcasting)
    mu_scale = scheduler.mu_t_0T_scale(t)  # Shape: (Batch, 1)
    sigma_scale = scheduler.sigma_t_0T(t)  # Shape: (Batch, 1)
    sigma_t = scheduler.sigma_t(t)  # Shape: (Batch, 1)
    dt = diffusion.dt  # Shape: (Batch, 1)

    # a_t = mu(t) * a_T + sigma(t) * noise
    # a_t = mu_scale * a_T_unsquashed + sigma_scale * noise

    a_t = a_0 + mu_scale * (action - a_0) + noise * sigma_scale

    # 4. Evaluate the policy \pi(a_t, t, o)
    # In your code structure, this is the forward model inside the diffusion class
    # It usually predicts the score or the noise.

    ctrl = sigma_t * jax.vmap(diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, contexts, t)
    old_ctrl = sigma_t * jax.vmap(old_diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, contexts, t)

    # Compute importance weights
    log_importance_weights = log_weights + Q_value.reshape(log_weights.shape)
    lm = actor_model.optimize_lm(log_importance_weights)
    smoothing = 1. / (1. + lm)
    smoothed_log_importance_weights = smoothing * log_importance_weights
    self_normalized_weights = jnp.exp(smoothed_log_importance_weights - logsumexp(smoothed_log_importance_weights))

    # 5. Compute MSE Loss: || \pi(a_t,t,o) - nabla_{a_T} Q(a_T, o) ||^2
    # We average over the feature dimension (-1) and then over the batch
    adjoint_state =  - Q_score
    adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl + sigma_t * adjoint_state), axis=-1) * dt

    lagrangian = actor_model.lagrangian()
    kl_scale = jax.lax.stop_gradient(lagrangian)

    kl_loss = jnp.mean(0.5 * jnp.mean(jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)))

    # The following penalty makes the KL loss more reactive
    # rho = 5.0  # Stiffness hyperparameter
    # violation = jax.nn.relu(kl_loss - cfg.kl_bound)  # effectively max(0, x)
    # quadratic_penalty = (rho / 2.0) * jnp.square(violation)

    weighted_adjoint_loss = jnp.sum(self_normalized_weights.reshape(adjoint_loss.shape) * adjoint_loss)
    # weighted_adjoint_loss = jnp.mean(adjoint_loss)

    actor_loss = weighted_adjoint_loss + kl_scale * kl_loss  # + quadratic_penalty
    # actor_loss = weighted_adjoint_loss #+ kl_scale * kl_loss  # + quadratic_penalty

    # # SAC target entropy loss
    # _, _, _, entropy, *_ = actor_model.sde_sample_and_kl(key, minibatch.obs, actor_target_model, stop_grad=True)
    # entropy = log_weights.mean()
    #
    # # Lagrangian constraint (follows temperature update)
    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - kl_bound)
    lagrangian_loss = lagrangian_loss.mean()
    #
    # total loss
    loss = jnp.mean(actor_loss)

    if update_kl_lagrangian:
        loss += lagrangian_loss

    # log diffusion coefficient (detached for safe logging)
    ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1)) * dt
    old_ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1)) * dt
    nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(Q_score), axis=-1))
    nabla_p_T_ref_grad_norm = jnp.mean(jnp.sum(jnp.square(-action / scheduler.sigma_T_0() ** 2), axis=-1))
    adjoint_norm = jnp.mean(jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))
    weighted_adjoint_norm = jnp.mean(
        self_normalized_weights.reshape(adjoint_loss.shape) * jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))

    metrics = dict(
        actor_loss=actor_loss,
        loss=loss,
        temp=actor_model.temperature(),
        abs_batch_action=jnp.abs(action).mean(),
        action_norm=jnp.mean(jnp.sum(jnp.square(action), axis=-1)),
        adjoint_norm=adjoint_norm,
        weighted_adjoint_norm=weighted_adjoint_norm,
        nabla_p_T_ref_grad_norm=nabla_p_T_ref_grad_norm,
        reward_mean=Q_value.mean(),
        kl=kl_loss,
        kl_loss=kl_loss,
        scaled_kl_loss=kl_scale * kl_loss,
        adjoint_loss=weighted_adjoint_loss,
        loss_ratio=weighted_adjoint_loss / (kl_scale * kl_loss),
        ctrl_norm=ctrl_norm,
        old_ctrl_norm=old_ctrl_norm,
        nabla_Q_norm=nabla_Q_norm,
        ESS=compute_reverse_ess(log_importance_weights),
        smoothed_ESS=compute_reverse_ess(smoothed_log_importance_weights),
        m_step_lagrangian_loss=lagrangian_loss,
        m_step_lagrangian=lagrangian,
        e_step_lagrangian=lm,
        entropy=jnp.mean(log_weights),
        # entropy_loss=target_entropy_loss,
        entropy_temp=actor_model.temperature(),
        log_path_weight_deterministic=log_path_weight_deterministic.mean(),
        log_path_weight_stochastic=log_path_weight_stochastic.mean(),
        log_p_T_ref_weight=log_p_T_ref.mean(),
    )
    return loss, metrics

def adjoint_matching_entr(key, params, contexts, train_state, action, prior_action, log_weights, log_path_weight_deterministic,
                          log_path_weight_stochastic, log_p_T_ref, target_log_prob, get_Q, batch_repetitions, update_kl_lagrangian, kl_bound, entr_bound):
    actor_model = nnx.merge(train_state.actor.graphdef, params)
    actor_target_model = nnx.merge(
        train_state.actor_target.graphdef, train_state.actor_target.params
    )
    # Access the diffusion components
    diffusion = actor_model.diffusion_model
    old_diffusion = actor_target_model.diffusion_model
    scheduler = diffusion.noise_scheduler

    batch_size = contexts.shape[0] * batch_repetitions

    key, act_key = jax.random.split(key, 2)
    key, act_key, act_rndm, norm_act_rndm = jax.random.split(key, 4)

    # action, prior_action, log_weights, log_path_weight_deterministic, log_path_weight_stochastic, log_p_T_ref = actor_model.sde_sample(
    #     act_key, contexts, stop_grad=True)

    a_0 = jnp.repeat(prior_action, batch_repetitions, axis=0)
    action = jnp.repeat(action, batch_repetitions, axis=0)
    contexts = jnp.repeat(contexts, batch_repetitions, axis=0)

    # actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
    # _, _, log_weights_entr_calc, _, _, _ = actor_model.sde_sample(act_key, contexts, stop_grad=True)
    # #
    # # a_0 = prior_action


    # Q_value, Q_score = jax.vmap(jax.value_and_grad(lambda x, c: target_log_prob(x, c).squeeze()), in_axes=(0, 0))(action, contexts)
    Q_value, Q_score = jax.vmap(jax.value_and_grad(lambda x, c: get_Q(x, c).squeeze()), in_axes=(0, 0))(action, contexts)


    #### Normalization Constant
    # we need to sample several actions per context, ideally uniform to have a better approx
    norm_acts = jax.random.uniform(norm_act_rndm, (contexts.shape[0], 1000), minval=-10, maxval=10)
    # Q_valsZ = target_log_prob(norm_acts, contexts)
    Q_valsZ = get_Q(norm_acts, contexts)

    Q_value = Q_value.reshape(Q_score.shape)
    log_weights = jnp.repeat(log_weights, batch_repetitions, axis=0)
    log_p_T_ref = jnp.repeat(log_p_T_ref, batch_repetitions, axis=0)

    # 1. Randomly sample time t between [0,1]
    key_t, key_noise = jax.random.split(key, 2)  # Split the key passed to actor_loss
    t = jax.random.uniform(key_t, (batch_size, 1))

    # 2. Randomly sample noise from N(0, I)
    noise = jax.random.normal(key_noise, action.shape)

    # 3. Sample a_t (Forward Diffusion Process)
    # a_T is the clean action from the buffer (minibatch.action)

    # Retrieve coefficients from scheduler (handling broadcasting)
    mu_scale = scheduler.mu_t_0T_scale(t)  # Shape: (Batch, 1)
    sigma_scale = scheduler.sigma_t_0T(t)  # Shape: (Batch, 1)
    sigma_t = scheduler.sigma_t(t)  # Shape: (Batch, 1)
    dt = diffusion.dt  # Shape: (Batch, 1)

    # a_t = mu(t) * a_T + sigma(t) * noise
    # a_t = mu_scale * a_T_unsquashed + sigma_scale * noise

    a_t = a_0 + mu_scale * (action - a_0) + noise * sigma_scale

    # 4. Evaluate the policy \pi(a_t, t, o)
    # In your code structure, this is the forward model inside the diffusion class
    # It usually predicts the score or the noise.

    ctrl = sigma_t * jax.vmap(diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, contexts, t)
    old_ctrl = sigma_t * jax.vmap(old_diffusion.fwd_model, in_axes=(0, 0, 0))(a_t, contexts, t)

    # Compute importance weights
    log_importance_weights = log_weights + Q_value.reshape(log_weights.shape)
    H_0 = 0.0
    # old_entropy = -0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1) + diffusion.prior_log_prob(a_0))
    old_entropy = log_weights.sum(axis=1).mean()
    # log_weights_entr_calc = jax.lax.stop_gradient(log_weights_entr_calc)
    # old_entropy = log_weights_entr_calc.sum(axis=1).mean()
    kappa = (old_entropy - H_0) * 0.99 + H_0
    # kappa = 0.001 - old_entropy

    # kappa = 1.0

    # jax.debug.print(
    #     "entropy_bound={}",
    #     kappa
    # )

    lm_tr, lm_entr, opt_lm_ent_lb, dual_val, state = actor_model.optimize_lm(Q_vals=Q_value, log_w=log_weights,
                                                                             log_p0=log_p_T_ref, ent_bound=-kappa,
                                                                             ent_lb_bound=-entr_bound)

    # smoothing = lm_tr / (1. + lm_tr + lm_entr)
    smoothed_log_importance_weights = (((1 + lm_entr + opt_lm_ent_lb) / (1. + lm_tr + lm_entr + opt_lm_ent_lb))*log_weights
                                       + (1. / (1. + lm_tr + lm_entr + opt_lm_ent_lb)) * Q_value.reshape(log_weights.shape))
    self_normalized_weights = jnp.exp(smoothed_log_importance_weights - logsumexp(smoothed_log_importance_weights))

    # 5. Compute MSE Loss: || \pi(a_t,t,o) - nabla_{a_T} Q(a_T, o) ||^2
    # We average over the feature dimension (-1) and then over the batch
    # adjoint_state =  - Q_score/(lm_entr + opt_lm_ent_lb + 1e-6)
    # adjoint_state =  - Q_score/(1e8) # TODO
    adjoint_state =  - Q_score
    scaler = (lm_entr + opt_lm_ent_lb)
    adjoint_loss = 0.5 * jnp.sum(jnp.square(ctrl * scaler + sigma_t * adjoint_state), axis=-1) * dt

    lagrangian = actor_model.lagrangian()
    kl_scale = jax.lax.stop_gradient(lagrangian)

    kl_loss = jnp.mean(0.5 * jnp.mean(jnp.sum(jnp.square(ctrl - old_ctrl), axis=-1)))

    # The following penalty makes the KL loss more reactive
    # rho = 5.0  # Stiffness hyperparameter
    # violation = jax.nn.relu(kl_loss - cfg.kl_bound)  # effectively max(0, x)
    # quadratic_penalty = (rho / 2.0) * jnp.square(violation)

    weighted_adjoint_loss = jnp.sum(self_normalized_weights.reshape(adjoint_loss.shape) * adjoint_loss)
    # weighted_adjoint_loss = jnp.mean(adjoint_loss)

    actor_loss = weighted_adjoint_loss + kl_scale * kl_loss  # + quadratic_penalty
    # actor_loss = weighted_adjoint_loss #+ kl_scale * kl_loss  # + quadratic_penalty

    # # SAC target entropy loss
    # _, _, _, entropy, *_ = actor_model.sde_sample_and_kl(key, minibatch.obs, actor_target_model, stop_grad=True)
    # entropy = log_weights.mean()
    #
    # # Lagrangian constraint (follows temperature update)
    lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_loss - kl_bound)
    lagrangian_loss = lagrangian_loss.mean()
    #
    # total loss
    loss = jnp.mean(actor_loss)

    if update_kl_lagrangian:
        loss += lagrangian_loss

    # log_opt_distr = log_weights + (1. / (lm_entr+1e-6)) * Q_value.reshape(log_weights.shape)
    # logZ = logsumexp((1. / (lm_entr+1e-6)) * Q_value.reshape(log_weights.shape)) - jnp.log(log_opt_distr.shape[0])
    # logQStar = (1. / (lm_entr+1e-6)) * Q_value.reshape(log_weights.shape) - logZ
    #
    # log_iw = log_opt_distr - logsumexp(log_opt_distr) - jnp.log(log_opt_distr.shape[0]) - logZ
    # iw = jnp.exp(log_iw)
    #
    # # entropy_new = jnp.sum(logQStar*iw)
    # entropy_new = jnp.sum(logQStar)

    # entropy_new = compute_entropy_via_importance_sampling(log_weights, Q_value,1 + lm_entr + opt_lm_ent_lb + lm_tr)
    entropy_new = compute_entropy_via_importance_sampling(log_weights, Q_value, 1 + lm_entr + opt_lm_ent_lb)
    # entropy_new = compute_entropy_via_importance_sampling(log_weights, Q_value, 1e8)
    entropy_unsmoothed = compute_entropy_via_importance_sampling(log_weights, Q_value, 1.0)
    # entropy_new = entropy_unsmoothed
    # entropy_new = compute_entropy_via_importance_sampling(log_weights, Q_value, 0.1)

    # entropy_new = compute_entropy_via_importance_sampling(lm_entr, Q_valsZ=Q_valsZ)
    # entropy_new = compute_entropy_via_importance_sampling(10.0, Q_valsZ=Q_valsZ)
    # entropy_unsmoothed = compute_entropy_via_importance_sampling(1.0, Q_valsZ=Q_valsZ)
    # _ = compute_entropy_via_importance_sampling(0.1, Q_valsZ=Q_valsZ)

    # log diffusion coefficient (detached for safe logging)
    ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1)) * dt
    old_ctrl_norm = 0.5 * jnp.mean(jnp.sum(jnp.square(old_ctrl), axis=-1)) * dt
    nabla_Q_norm = jnp.mean(jnp.sum(jnp.square(Q_score), axis=-1))
    nabla_p_T_ref_grad_norm = jnp.mean(jnp.sum(jnp.square(-action / scheduler.sigma_T_0() ** 2), axis=-1))
    adjoint_norm = jnp.mean(jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))
    weighted_adjoint_norm = jnp.mean(
        self_normalized_weights.reshape(adjoint_loss.shape) * jnp.sum(jnp.square(sigma_t * adjoint_state), axis=-1))

    # jax.debug.print(
    #     "#######################################################################"
    #     "entropy={}",
    #     entropy_new,
    # )


    metrics = dict(
        actor_loss=actor_loss,
        loss=loss,
        temp=actor_model.temperature(),
        abs_batch_action=jnp.abs(action).mean(),
        action_norm=jnp.mean(jnp.sum(jnp.square(action), axis=-1)),
        adjoint_norm=adjoint_norm,
        weighted_adjoint_norm=weighted_adjoint_norm,
        nabla_p_T_ref_grad_norm=nabla_p_T_ref_grad_norm,
        reward_mean=Q_value.mean(),
        kl=kl_loss,
        kl_loss=kl_loss,
        scaled_kl_loss=kl_scale * kl_loss,
        adjoint_loss=weighted_adjoint_loss,
        loss_ratio=weighted_adjoint_loss / (kl_scale * kl_loss),
        ctrl_norm=ctrl_norm,
        old_ctrl_norm=old_ctrl_norm,
        nabla_Q_norm=nabla_Q_norm,
        ESS=compute_reverse_ess(log_importance_weights),
        smoothed_ESS=compute_reverse_ess(smoothed_log_importance_weights),
        m_step_lagrangian_loss=lagrangian_loss,
        m_step_lagrangian=lagrangian,
        e_step_lagrangian=lm_tr,
        e_step_lagrangian_entr=lm_entr,
        e_step_lagrangian_lb_entr=opt_lm_ent_lb,
        e_step_dual_val=dual_val,
        # entropy=jnp.mean(((1+ lm_entr) / (1. + lm_tr + lm_entr))*log_weights),
        entropy=entropy_new,
        entropy_unsmoothed=entropy_unsmoothed,
        # logZ=logZ,
        # entropy=-0.5 * jnp.mean(jnp.sum(jnp.square(ctrl), axis=-1) + diffusion.prior_log_prob(a_0)),
        old_entropy = old_entropy,
        entr_bound=kappa,
        # entropy_loss=target_entropy_loss,
        entropy_temp=actor_model.temperature(),
        log_path_weight_deterministic=log_path_weight_deterministic.mean(),
        log_path_weight_stochastic=log_path_weight_stochastic.mean(),
        log_p_T_ref_weight=log_p_T_ref.mean(),
        lbfgs_grad=state.grad,
        lbfgs_num_updates=state.num_updates,
        lbfgs_error=state.error,
        log_weights=log_weights.mean()
    )
    return loss, metrics

#
import jax.numpy as jnp
# from jax.scipy.special import logsumexp
def compute_entropy_via_importance_sampling(log_weights, Q_value, lm_entr):
    """
    Args:
        log_weights: Log probabilities of the reference distribution (part of q*).
        log_p: Log probabilities of the samples under the proposal distribution p.
        Q_value: Q-values associated with the samples.
        lm_entr: The temperature/entropy coefficient (lambda).
    """
    # 1. Calculate unnormalized target log-probabilities: log(q_tilde)
    # q*(x) = exp(Q(x)/lambda) / Z

    # jax.debug.print(
    #     "sum_expQ={}",
    #     jnp.
    #     jnp.sum(jnp.exp(Q_value - logsumexp(Q_value) + jnp.log(Q_value.shape[0])))
    # )

    T = lm_entr
    log_q_tilde = Q_value.reshape(log_weights.shape)/(T+1e-6)
    # 2. Calculate unnormalized importance weights: log(w) = log(q_tilde) - log(p)
    # This represents the ratio q*(x) / p(x) without the unknown Z constant
    log_importance_weights = log_q_tilde + log_weights
    # 3. Estimate the Log Partition Function (Log Z)
    # Z = E_p[q_tilde(x) / p(x)] approx (1/N) * sum(exp(log_importance_weights))
    N = log_q_tilde.shape[0]
    log_Z = logsumexp(log_importance_weights) - jnp.log(N)
    # 4. Calculate Normalized Importance Weights
    # w_norm = softmax(log_importance_weights)
    norm_weights = jax.nn.softmax(log_importance_weights.squeeze()).reshape(log_importance_weights.shape)
    # 5. Compute Entropy
    # H(q*) = - E_q [log q*(x)]
    #       = - E_q [log q_tilde(x) - log Z]
    #       = - sum(w_norm * log_q_tilde) + log_Z
    entropy = -jnp.sum(norm_weights * log_q_tilde) + log_Z
    # jax.debug.print(
    #     "entropy={}, real_entropy={}, temperature={}",
    #     entropy,
    #     0.5 * jnp.log(2.0 * jnp.pi * jnp.e * 1.0 ** 2),
    #     T
    # )

    # jax.debug.print(
    #     "sum_norm_weights={}",
    #     jnp.sum(norm_weights)
    # )
    return entropy




# import jax.numpy as jnp
# # from jax.scipy.special import logsumexp
# def compute_entropy_via_importance_sampling(log_weights, Q_value, lm_entr, Q_valsZ):
#     """
#     Args:
#         log_weights: Log probabilities of the reference distribution (part of q*).
#         log_p: Log probabilities of the samples under the proposal distribution p.
#         Q_value: Q-values associated with the samples.
#         lm_entr: The temperature/entropy coefficient (lambda).
#     """
#     # 1. Calculate unnormalized target log-probabilities: log(q_tilde)
#     # q*(x) = exp(Q(x)/lambda) / Z
#
#     # jax.debug.print(
#     #     "sum_expQ={}",
#     #     jnp.
#     #     jnp.sum(jnp.exp(Q_value - logsumexp(Q_value) + jnp.log(Q_value.shape[0])))
#     # )
#
#     T = lm_entr
#     log_q_tilde = Q_value.reshape(log_weights.shape)/T
#     # 2. Calculate unnormalized importance weights: log(w) = log(q_tilde) - log(p)
#     # This represents the ratio q*(x) / p(x) without the unknown Z constant
#     log_importance_weights = log_q_tilde + log_weights
#     # 3. Estimate the Log Partition Function (Log Z)
#     # Z = E_p[q_tilde(x) / p(x)] approx (1/N) * sum(exp(log_importance_weights))
#     N = log_q_tilde.shape[0]
#     log_Z = logsumexp(log_importance_weights) - jnp.log(N)
#     # 4. Calculate Normalized Importance Weights
#     # w_norm = softmax(log_importance_weights)
#     norm_weights = jax.nn.softmax(log_importance_weights.squeeze()).reshape(log_importance_weights.shape)
#     # 5. Compute Entropy
#     # H(q*) = - E_q [log q*(x)]
#     #       = - E_q [log q_tilde(x) - log Z]
#     #       = - sum(w_norm * log_q_tilde) + log_Z
#     entropy = -jnp.sum(norm_weights * log_q_tilde) + log_Z
#     jax.debug.print(
#         "entropy={}, real_entropy={}, temperature={}",
#         entropy,
#         0.5 * jnp.log(2.0 * jnp.pi * jnp.e * 1.0 ** 2),
#         T
#     )
#
#     # jax.debug.print(
#     #     "sum_norm_weights={}",
#     #     jnp.sum(norm_weights)
#     # )
#     return entropy




# def compute_entropy_via_importance_sampling(lm_entr, Q_valsZ):
#     """
#     Args:
#         log_weights: Log probabilities of the reference distribution (part of q*).
#         log_p: Log probabilities of the samples under the proposal distribution p.
#         Q_value: Q-values associated with the samples.
#         lm_entr: The temperature/entropy coefficient (lambda).
#     """
#     T = lm_entr
#     log_q_tildeZ = Q_valsZ/T
#     logZpc = logsumexp(log_q_tildeZ, axis=1) - jnp.log(log_q_tildeZ.shape[1])
#
#     log_weights = log_q_tildeZ # + jnp.log(log_q_tildeZ.shape[1])
#     norm_weights = jax.nn.softmax(log_weights, axis=1)
#     # 5. Compute Entropy
#     # H(q*) = - E_q [log q*(x)]
#     #       = - E_q [log q_tilde(x) - log Z]
#     #       = - sum(w_norm * log_q_tilde) + log_Z
#     entropy = -jnp.sum(norm_weights * log_q_tildeZ, axis=1) + logZpc
#     entropy = entropy.mean()
#     real_entropy = 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * 1.0 ** 2* lm_entr)
#     jax.debug.print(
#         "entropy={}, real_entropy={}, temperature={}",
#         entropy, real_entropy, T
#     )
#     # jax.debug.print(
#     #     "sum_norm_weights={}",
#     #     jnp.sum(norm_weights)
#     # )
#     return entropy