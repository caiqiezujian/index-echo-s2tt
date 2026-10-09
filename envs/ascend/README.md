# Ascend 状态

当前交付针对单张 NVIDIA 5090。Ascend 910B 的设备 provider、算子回归和真实模型验收尚未实现，状态为 BLOCKED。

后续必须独立核对 TorchNPU/CANN/驱动/固件/CPU 架构，建立环境锁并按音频特征→音频塔→connector→decoder 对照。不要把 CUDA 环境锁直接用于 NPU，或全仓替换设备字符串。
