# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Record DiT forward device events without synchronizing between steps."""
import importlib.abc
import importlib.machinery
import json
import sys

_active = None


def before_forward(module, args):
    if _active is None:
        return
    import torch
    start, stop = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    start.record(torch.npu.current_stream())
    _active['pairs'].append((_active['step'] + 1, start, stop))


def after_forward(module, args, output):
    if _active is not None:
        import torch
        _active['pairs'][-1][2].record(torch.npu.current_stream())


def patch_sampling(module):
    original = module.MiniMaxH3ModelBatch.__call__

    def measured(self, evaluations):
        if _active is not None and evaluations:
            if len(evaluations) != 1:
                raise RuntimeError('DiT timing observer requires a single request')
            transformer = self.models[evaluations[0].state].transformer
            if not hasattr(transformer, '_h3_dit_timing_handles'):
                transformer._h3_dit_timing_handles = (
                    transformer.register_forward_pre_hook(before_forward, prepend=True),
                    transformer.register_forward_hook(after_forward),
                )
            _active['step'] = evaluations[0].state.step_index
        return original(self, evaluations)

    module.MiniMaxH3ModelBatch.__call__ = measured


def patch_loop(module):
    original = module.minimax_h3_denoise_loop

    def measured(*args, **kwargs):
        global _active
        if _active is not None:
            raise RuntimeError('Nested denoise loops are not supported by this observer')
        _active = {'pairs': [], 'step': 0}
        try:
            output = original(*args, **kwargs)
            pairs = _active['pairs']
            expected = len(kwargs['sigmas_video']) - 1
            if [item[0] for item in pairs] != list(range(1, expected + 1)):
                raise RuntimeError('Missing or duplicate transformer timing intervals')
            pairs[-1][2].synchronize()
            import torch.distributed as dist
            values = [{'step': step, 'device_ms': start.elapsed_time(stop)} for step, start, stop in pairs]
            record = {'rank': dist.get_rank(), 'metric': 'DiT transformer forward device events',
                      'steps': values, 'dit_forward_total_ms': sum(item['device_ms'] for item in values)}
            print('[H3_DIT_EVENTS] ' + json.dumps(record), flush=True)
            return output
        finally:
            _active = None

    module.minimax_h3_denoise_loop = measured


class Loader(importlib.abc.Loader):
    def __init__(self, original, patch):
        self.original, self.patch = original, patch

    def create_module(self, spec):
        return self.original.create_module(spec)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def exec_module(self, module):
        self.original.exec_module(module)
        self.patch(module)


class Finder(importlib.abc.MetaPathFinder):
    patches = {'vllm_omni.diffusion.models.minimax_h3.sampling': patch_sampling,
               'vllm_omni.diffusion.models.minimax_h3.denoise_loop': patch_loop}

    def find_spec(self, fullname, path=None, target=None):
        patch = self.patches.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise RuntimeError('Cannot locate measured H3 module')
        spec.loader = Loader(spec.loader, patch)
        return spec


def install():
    sys.meta_path.insert(0, Finder())
