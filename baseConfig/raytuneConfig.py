from ray import train


scaling_config = train.ScalingConfig(
    num_workers=4,
    resources_per_worker={
        "CPU": 4,
        "GPU": 0,
    },
    use_gpu=False,
)