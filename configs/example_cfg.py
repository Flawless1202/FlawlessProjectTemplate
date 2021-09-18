name = "ExampleModelExp"

hparams = dict(
    hparam1="hparam1",
    hparam2="hparam2",
    hparam3="hparam3",
    hparam4="hparam4",
)

model = dict(
    type="ModelType",
    embedding=dict(type="EmbeddingType", **kwargs),
    encoder=dict(type="EncoderType", **kwargs),
    decoder=dict(type="DecoderType", **kwargs),
    loss=dict(type="LossType"),
)
metrics = dict(type="MetricsType")

dataset = "DatasetType"
data = dict(
    batch_size=16,
    num_workers=24,
    train=dict(type=dataset, **kwargs),
    val=dict(type=dataset, **kwargs),
    test=dict(type=dataset, **kwargs),
)
# training and testing settings
train_cfg = dict(
    loss=dict(),
)
val_cfg = dict(
    loss=dict(),
    predict=dict(),
    target=dict(),
)
test_cfg = dict(
    predict=dict(),
    target=dict(),
)

# optimizer
optimizer_cfg = dict(type="OptimizerType", **kwargs)
scheduler_cfg = dict(type="SchedulerType", **kwargs)
warm_up_cfg = dict(type="WarmUpType", **kwargs)

runner_cfg = dict(
    random_seed=123456,
    num_gpus=1,
    max_epochs=10,
    gradient_clip_val=0.5,
    precision=32,
    monitor_metrics="some_metrics",
    checkpoint_monitor_metrics=(
        "metrics used to monitor checkpoint save",
        ("min" or "max"),
    ),
    resume_from_checkpoint=None,
    accumulate_grad_batches=1,
    simple_profiler=False,
    check_val_every_n_epoch=1,
    test_checkpoint=None,
)
