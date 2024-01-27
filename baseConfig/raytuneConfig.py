from ray import train

# 一般来说，workers的数量不应超过cpu的数量。如果workers的数量超过了cpu的数量，可能会导致资源竞争和性能下降。
# 建议将workers设置为小于等于cpu的数量，以充分利用可用的CPU资源并确保高效的并行训练。


scaling_config = train.ScalingConfig(
    num_workers=4,
    resources_per_worker={
        "CPU": 4,
        "GPU": 0,
    },
    use_gpu=False,
)