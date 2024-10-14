from ray import train

# 一般来说，workers的数量不应超过cpu的数量。如果workers的数量超过了cpu的数量，可能会导致资源竞争和性能下降。

# trainer_resources: 分配给 traner 的所有资源
# num_workers：并行的进程数量
# resources_per_worker：每个进程平均获得资源数量

scaling_config = train.ScalingConfig(
    trainer_resources={'CPU': 0, 'GPU': 1},
    num_workers=1,
    resources_per_worker={
        "CPU": 0,
        "GPU": 1,
    },
    use_gpu=True,
)