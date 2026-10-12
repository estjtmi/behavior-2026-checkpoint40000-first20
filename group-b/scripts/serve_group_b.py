"""Serve unchanged B1K inference code using official 2026 transport and indices.

The training repo uses the old omnigibson.learning import paths. Only the
proprioception index constant and websocket transport are needed at inference.
Load those from the pinned official source without importing the simulator into
the isolated policy environment. No evaluator or model parameters are altered.
"""
import ast
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
import sys
import types
from collections import OrderedDict

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.45')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np

OFFICIAL = Path('/workspace/BEHAVIOR-1K/OmniGibson/omnigibson')


def prepare_legacy_imports():
    for name in ['omnigibson', 'omnigibson.learning', 'omnigibson.learning.utils']:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    source = OFFICIAL / 'eval/utils/eval_utils.py'
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'PROPRIOCEPTION_INDICES'
                     for target in node.targets)]
    assert len(nodes) == 1
    values = {'np': np, 'OrderedDict': OrderedDict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), values)
    indices = types.ModuleType('omnigibson.learning.utils.eval_utils')
    indices.PROPRIOCEPTION_INDICES = values['PROPRIOCEPTION_INDICES']
    sys.modules[indices.__name__] = indices
    macros = types.ModuleType('omnigibson.macros')
    macros.gm = types.SimpleNamespace(DEBUG=True)
    sys.modules[macros.__name__] = macros
    source = OFFICIAL / 'eval/utils/network_utils.py'
    spec = importlib.util.spec_from_file_location('group_b_official_transport', source)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    logging.info('Using unchanged official transport %s sha256=%s', source,
                 hashlib.sha256(source.read_bytes()).hexdigest())
    return transport


def main():
    logging.basicConfig(level=logging.INFO)
    transport = prepare_legacy_imports()
    from b1k.policies import policy_config
    from b1k.training import config
    from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper
    policy = policy_config.create_trained_policy(
        config.get_config('pi_behavior_b1k_fast'),
        '/workspace/checkpoints/behavior-2026-pi05-15tasks/checkpoint_40000',
        sample_kwargs={'num_steps': 20},
    )
    class SingleEnvironmentWrapper(B1KPolicyWrapper):
        """Remove the evaluator's singleton environment axis at the boundary.

        The original rolling-inpainting transforms expect unbatched raw state
        and images alongside the retained (4, 23) action array. The original
        PiBehaviorPolicy adds the model batch axis after those transforms.
        """
        def act(self, observation):
            observation = dict(observation)
            state = np.asarray(observation['robot_r1::proprio'])
            if state.ndim == 2:
                assert state.shape[0] == 1
                observation['robot_r1::proprio'] = state[0]
            for sensor in ['zed_link', 'left_realsense_link', 'right_realsense_link']:
                key = f'robot_r1::robot_r1:{sensor}:Camera:0::rgb'
                value = np.asarray(observation[key])
                if value.ndim == 4:
                    assert value.shape[0] == 1
                    value = value[0]
                assert value.shape[:2] == (224, 224), (key, value.shape)
                observation[key] = value
            assert np.asarray(observation['robot_r1::proprio']).shape == (61,)
            return super().act(observation)

    wrapped = SingleEnvironmentWrapper(policy, task_id=7)
    rng_before_preflight = policy._rng
    obs = {'robot_r1::proprio': np.zeros((1, 61), dtype=np.float32),
           'task_id': np.array([7], dtype=np.int64)}
    for sensor in ['zed_link', 'left_realsense_link', 'right_realsense_link']:
        obs[f'robot_r1::robot_r1:{sensor}:Camera:0::rgb'] = np.zeros((1, 224, 224, 4), dtype=np.uint8)
    action = wrapped.act(obs).numpy()
    assert action.shape == (23,) and np.isfinite(action).all(), action
    # Exercise rolling inpainting as well as the first action horizon.
    for _ in range(21):
        action = wrapped.act(obs).numpy()
        assert action.shape == (23,) and np.isfinite(action).all()
    wrapped.reset()
    policy._rng = rng_before_preflight
    logging.info('Policy input/action preflight passed, including rolling inpainting.')
    transport.WebsocketPolicyServer(
        wrapped, host='127.0.0.1', port=8010,
        metadata={'checkpoint': 'estjtmi/behavior-2026-pi05-15tasks/checkpoint_40000',
                  'revision': '479fd5240a5150d69903506d01c84cd77d25bbd2',
                  'group': 'B', 'camera_resolution': [224, 224]},
    ).serve_forever()


if __name__ == '__main__':
    main()
