from ray import train


scaling_config = train.ScalingConfig(
    num_workers=8,
    resources_per_worker={
        "CPU": 8,
        "GPU": 0,
    },
    use_gpu=False,
)