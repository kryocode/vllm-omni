"""Matched NPU process initialization for both latest-main measurement arms."""
import runpy
import torch
import torch_npu

torch.npu.config.allow_internal_format = True

if __name__ == '__main__':
    runpy.run_module('vllm.entrypoints.cli.main', run_name='__main__', alter_sys=True)
