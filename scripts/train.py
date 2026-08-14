import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)

    flat_expected = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    for kp, v in flat_loaded.items():
        if kp not in flat_expected:
            raise ValueError(f"Loaded unexpected parameter at {jax.tree_util.keystr(kp)}")
        expected = flat_expected[kp]
        if expected.shape != v.shape:
            raise ValueError(
                f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {expected.shape}, got {v.shape}"
            )
        if expected.dtype != v.dtype:
            raise ValueError(
                f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {expected.dtype}, got {v.dtype}"
            )

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _inherit_eval_data_overrides(
    train_data: _config.DataConfigFactory, eval_data: _config.DataConfigFactory
) -> _config.DataConfigFactory:
    """Apply user overrides from train data to eval data, preserving eval split."""
    if type(train_data) is not type(eval_data):
        return eval_data

    eval_field_names = {field.name for field in dataclasses.fields(eval_data)}
    overrides = {}
    for field in dataclasses.fields(train_data):
        if field.name in ("split", "split_jsonl") or field.name not in eval_field_names:
            continue
        overrides[field.name] = getattr(train_data, field.name)
    return dataclasses.replace(eval_data, **overrides)


def _next_eval_batch(eval_loader, eval_iter):
    try:
        return next(eval_iter), eval_iter
    except StopIteration:
        eval_iter = iter(eval_loader)
        return next(eval_iter), eval_iter


_MODULE_LOG_FILTERS = (
    ("status_encoder", nnx_utils.PathRegex("status_encoder/.*")),
    ("done_head", nnx_utils.PathRegex("done_head/.*")),
    ("dynamics_encoder", nnx_utils.PathRegex("dynamics_encoder/.*")),
    ("latent_steering", nnx_utils.PathRegex("latent_steering/.*")),
    ("cosmos_time_embed", nnx_utils.PathRegex("cosmos_time_embed/.*")),
    ("cosmos_view_embed", nnx_utils.PathRegex("cosmos_view_embed/.*")),
    # These filters cover the full pretrained streams. The *_lora metrics below
    # remain useful only for configs that explicitly use LoRA variants.
    ("pi05_vlm", nnx_utils.PathRegex(r"PaliGemma/(img/.*|llm/(?!.*_1).*)")),
    ("action_expert", nnx_utils.PathRegex(r"PaliGemma/llm/.*_1.*")),
    ("pi05_vlm_lora", nnx_utils.PathRegex("PaliGemma/llm/.*_0/.*lora.*")),
    ("action_expert_lora", nnx_utils.PathRegex("PaliGemma/llm/.*_1/.*lora.*")),
    ("action_in_proj", nnx_utils.PathRegex("action_in_proj/.*")),
    ("time_mlp", nnx_utils.PathRegex("time_mlp_(in|out)/.*")),
    ("action_out_proj", nnx_utils.PathRegex("action_out_proj/.*")),
    ("action_prior_encoder", nnx_utils.PathRegex("action_prior_encoder/.*")),
)


def _filtered_global_norm(state: nnx.State, filter_: nnx.filterlib.Filter) -> at.Array:
    filtered = state.filter(filter_)
    if not filtered.flat_state():
        return jnp.asarray(0.0, dtype=jnp.float32)
    return optax.global_norm(filtered)


def _select_sharding_tree(sharding_tree, value_tree):
    """Return a sharding pytree with the same structure as a possibly-partial value tree."""
    if isinstance(value_tree, dict):
        return {key: _select_sharding_tree(sharding_tree[key], value) for key, value in value_tree.items()}
    return sharding_tree


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    partial_params_sharding = _select_sharding_tree(state_sharding.params.to_pure_dict(), partial_params)

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=(replicated_sharding, partial_params_sharding),
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if getattr(config.model, "enable_status_head", False):
            if not isinstance(model, _model.BaseModel) or not hasattr(model, "compute_loss_with_status"):
                raise TypeError("Status training requires a Pi0/Pi05 model with compute_loss_with_status().")
            action_loss, done_loss, done_logit = model.compute_loss_with_status(
                rng,
                observation,
                actions,
                train=True,
                include_action_loss=not config.model.status_only_trainable,
            )
            action_loss_mean = jnp.mean(action_loss)
            if config.model.status_only_trainable:
                total_loss = config.model.status_loss_weight * done_loss
            else:
                total_loss = action_loss_mean + config.model.status_loss_weight * done_loss
            return total_loss, (action_loss_mean, done_loss, done_logit)
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), None

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "update_norm": optax.global_norm(updates),
        "param_norm": optax.global_norm(kernel_params),
    }
    if aux is not None:
        action_loss_mean, done_loss, done_logit = aux
        done_prob = jax.nn.sigmoid(done_logit.squeeze(-1))
        done_label = observation.done_label.astype(done_prob.dtype)
        done_pred = done_prob >= 0.5
        done_true = done_label >= 0.5
        positive = done_true.astype(jnp.float32)
        negative = 1.0 - positive
        correct = (done_pred == done_true).astype(jnp.float32)
        positive_count = jnp.maximum(jnp.sum(positive), 1.0)
        negative_count = jnp.maximum(jnp.sum(negative), 1.0)
        info.update(
            {
                "action_loss": action_loss_mean,
                "done_loss": done_loss,
                "done_accuracy": jnp.mean(correct),
                "done_positive_accuracy": jnp.sum(correct * positive) / positive_count,
                "done_negative_accuracy": jnp.sum(correct * negative) / negative_count,
                "done_prob_positive_mean": jnp.sum(done_prob * positive) / positive_count,
                "done_prob_negative_mean": jnp.sum(done_prob * negative) / negative_count,
            }
        )
    if getattr(config.model, "use_action_steering", False):
        # Reuse the exact split used by Pi05Cosmos.compute_loss so the logged
        # keep ratio describes the same sample-level dropout decision. This is
        # diagnostic-only; it is not part of the action loss.
        action_dropout_rng = jax.random.split(train_rng, 5)[4]
        action_prior_token, action_prior_mask = model.build_action_prior_token(
            observation, train=True, dropout_rng=action_dropout_rng
        )
        if action_prior_token is None or action_prior_mask is None:
            info["action_prior_keep_ratio"] = jnp.asarray(0.0, dtype=jnp.float32)
            info["action_prior_token_norm"] = jnp.asarray(0.0, dtype=jnp.float32)
        else:
            info["action_prior_keep_ratio"] = jnp.mean(action_prior_mask.astype(jnp.float32))
            info["action_prior_token_norm"] = jnp.mean(
                jnp.linalg.norm(action_prior_token.astype(jnp.float32), axis=-1)
            )
        if observation.wam_action_prior is not None:
            raw_prior = observation.wam_action_prior.astype(jnp.float32)
            info["wam_action_prior_mean"] = jnp.mean(raw_prior)
            info["wam_action_prior_std"] = jnp.std(raw_prior)
    for module_name, module_filter in _MODULE_LOG_FILTERS:
        info[f"grad_norm/{module_name}"] = _filtered_global_norm(grads, module_filter)
        info[f"update_norm/{module_name}"] = _filtered_global_norm(updates, module_filter)
        info[f"param_norm/{module_name}"] = _filtered_global_norm(new_params, module_filter)
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    params = state.ema_params if config.eval_use_ema and state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()

    observation, actions = batch
    if getattr(config.model, "enable_status_head", False):
        action_loss, done_loss, done_logit = model.compute_loss_with_status(
            rng,
            observation,
            actions,
            train=False,
            include_action_loss=not config.model.status_only_trainable,
        )
        done_prob = jax.nn.sigmoid(done_logit.squeeze(-1))
        done_label = observation.done_label.astype(done_prob.dtype)
        done_correct = ((done_prob >= 0.5) == (done_label >= 0.5)).astype(jnp.float32)
        return {
            "eval/loss": (
                config.model.status_loss_weight * done_loss
                if config.model.status_only_trainable
                else jnp.mean(action_loss) + config.model.status_loss_weight * done_loss
            ),
            "eval/action_loss": jnp.mean(action_loss),
            "eval/done_loss": done_loss,
            "eval/done_accuracy": jnp.mean(done_correct),
        }
    loss = jnp.mean(model.compute_loss(rng, observation, actions, train=False))
    return {"eval/loss": loss}


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        max_to_keep=config.max_checkpoints_to_keep,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    _device_images = {k: np.asarray(jax.device_get(v)) for k, v in batch[0].images.items()}
    images_to_log = [
        wandb.Image(np.concatenate([_device_images[k][i] for k in _device_images], axis=1))
        for i in range(min(5, len(next(iter(_device_images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        train_state = jax.device_put(train_state, train_state_sharding)
        jax.block_until_ready(train_state)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        functools.partial(eval_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    eval_loader = None
    eval_iter = None
    if config.eval_data is not None and config.eval_interval > 0:
        eval_data = _inherit_eval_data_overrides(config.data, config.eval_data)
        eval_config = dataclasses.replace(config, data=eval_data)
        eval_loader = _data_loader.create_data_loader(
            eval_config,
            sharding=data_sharding,
            shuffle=False,
            num_batches=config.eval_num_batches,
        )
        eval_iter = iter(eval_loader)
        logging.info("Initialized eval data loader for %d batches.", config.eval_num_batches)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        if eval_iter is not None and (step % config.eval_interval == 0 or step == config.num_train_steps - 1):
            eval_infos = []
            for eval_batch_idx in range(config.eval_num_batches):
                eval_batch, eval_iter = _next_eval_batch(eval_loader, eval_iter)
                eval_rng = jax.random.fold_in(train_rng, step * config.eval_num_batches + eval_batch_idx)
                with sharding.set_mesh(mesh):
                    eval_infos.append(peval_step(eval_rng, train_state, eval_batch))
            stacked_eval_infos = common_utils.stack_forest(eval_infos)
            reduced_eval_info = jax.device_get(jax.tree.map(jnp.mean, stacked_eval_infos))
            eval_info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_eval_info.items())
            pbar.write(f"Step {step}: {eval_info_str}")
            wandb.log(reduced_eval_info, step=step)
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
