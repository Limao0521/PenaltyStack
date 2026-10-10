"""PPO training plumbing: save/load of trained policies, the plan,
convergence detection, safety stops and the restart supervisor."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from penalty.rl.evaluation import evaluate

HAS_SB3 = importlib.util.find_spec('stable_baselines3') is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))


def _window(monitor, goals, errors=0):
    """Feed one full window with `goals` goals and `errors` sim errors."""
    entry = None
    for i in range(monitor.window):
        outcome = 'goal' if i < goals else 'sim_error' if i < goals + errors else 'blocked'
        entry = monitor.add({'outcome': outcome, 'fallen': outcome == 'sim_error'})
    return entry


@unittest.skipUnless(HAS_SB3, 'stable-baselines3 not installed')
class ConvergenceTests(unittest.TestCase):
    def test_converges_after_patience_windows_without_improvement(self):
        from penalty.rl.training import ConvergenceMonitor
        monitor = ConvergenceMonitor('striker', window=100, patience=3, min_seconds=0)
        for goals in (10, 30, 50):
            self.assertTrue(_window(monitor, goals)['improved'])
        for goals in (52, 49, 55):  # within one standard error of 0.50
            self.assertFalse(_window(monitor, goals)['improved'])
        self.assertEqual(monitor.best, .5)
        self.assertTrue(monitor.converged(elapsed_s=0))

    def test_improvement_resets_patience_and_minimum_time_holds(self):
        from penalty.rl.training import ConvergenceMonitor
        monitor = ConvergenceMonitor('striker', window=100, patience=2, min_seconds=3600)
        _window(monitor, 30)
        _window(monitor, 31)
        self.assertEqual(monitor.stale, 1)
        _window(monitor, 60)
        self.assertEqual(monitor.stale, 0)
        _window(monitor, 60)
        _window(monitor, 59)
        self.assertFalse(monitor.converged(elapsed_s=1800))
        self.assertTrue(monitor.converged(elapsed_s=3600))

    def test_state_survives_resume_and_detects_regression(self):
        from penalty.rl.training import ConvergenceMonitor
        monitor = ConvergenceMonitor('keeper', window=50, patience=4, min_seconds=0)
        for _ in range(50):
            monitor.add({'outcome': 'save', 'fallen': False})
        resumed = ConvergenceMonitor('keeper', 50, 4, 0, state=monitor.state())
        self.assertEqual(resumed.best, 1.)
        self.assertFalse(resumed.regressed())
        for _ in range(50):
            resumed.add({'outcome': 'goal', 'fallen': False})
        self.assertTrue(resumed.regressed())
        self.assertEqual(len(resumed.history), 2)

    def test_unhealthy_simulator_aborts(self):
        from penalty.rl.training import ConvergenceMonitor, TrainingAborted, check_health
        monitor = ConvergenceMonitor('striker', window=100, patience=3, min_seconds=0)
        check_health(_window(monitor, 40, errors=10))
        with self.assertRaises(TrainingAborted):
            check_health(_window(monitor, 40, errors=11))

    def test_acceptance_gates(self):
        from penalty.rl.training import acceptance_gates
        baseline = {'goal_rate': .2, 'any_fall_rate': .1}
        self.assertTrue(all(acceptance_gates('striker', {'goal_rate': .2, 'any_fall_rate': .15},
                                             baseline).values()))
        gates = acceptance_gates('striker', {'goal_rate': .19, 'any_fall_rate': .16}, baseline)
        self.assertEqual(gates, {'primary_ge_baseline': False, 'falls_le_baseline_plus_5pct': False})

    def test_watchdog_detects_stall(self):
        from penalty.rl.training import Watchdog
        watchdog = Watchdog(limit_s=10, exit_fn=lambda: None)
        self.assertFalse(watchdog.stalled(now=watchdog.last + 5))
        self.assertTrue(watchdog.stalled(now=watchdog.last + 11))

    def test_non_finite_weights_are_detected(self):
        import torch
        from stable_baselines3 import PPO
        from penalty.rl.training import build_env, finite_policy
        model = PPO('MlpPolicy', build_env('striker', 'synthetic', 'empty'), n_steps=64,
                    batch_size=64, device='cpu')
        self.assertTrue(finite_policy(model))
        with torch.no_grad():
            next(model.policy.parameters())[0].fill_(float('nan'))
        self.assertFalse(finite_policy(model))


@unittest.skipUnless(HAS_SB3, 'stable-baselines3 not installed')
class AsyncCollectionTests(unittest.TestCase):
    def _model(self, role='striker', opponent='empty', **kwargs):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from penalty.rl.async_ppo import AsyncPPO
        from penalty.rl.training import build_env
        venv = VecNormalize(DummyVecEnv([lambda: build_env(role, 'synthetic', opponent)]),
                            norm_reward=False, clip_obs=10.)
        settings = dict(n_steps=512, batch_size=128, n_epochs=4, learning_rate=3e-4, seed=0)
        settings.update(kwargs)
        return AsyncPPO('MlpPolicy', venv, device='cpu', policy_kwargs={'net_arch': [128, 128]},
                        **settings), venv

    def test_numpy_policy_matches_torch(self):
        import torch
        from penalty.rl.actors import NumpyPolicy
        from penalty.rl.async_ppo import export_policy
        model, venv = self._model('keeper', 'synthetic')
        with torch.no_grad():
            model.policy.log_std.fill_(-.3)
        policy = NumpyPolicy(export_policy(model.policy))
        rng = np.random.default_rng(0)
        obs = rng.normal(size=(16, 114)).astype(np.float32)
        actions = rng.normal(size=(16, 3)).astype(np.float32)
        with torch.no_grad():
            values, log_probs, _ = model.policy.evaluate_actions(torch.as_tensor(obs),
                                                                 torch.as_tensor(actions))
            means = model.policy.get_distribution(torch.as_tensor(obs)).distribution.mean
        for i in range(16):
            np.testing.assert_allclose(policy.mean(obs[i]), means[i].numpy(), atol=1e-5)
            self.assertAlmostEqual(policy.value(obs[i]), float(values[i]), places=4)
            self.assertAlmostEqual(policy.log_prob(policy.mean(obs[i]), actions[i]),
                                   float(log_probs[i]), places=3)
        venv.close()

    def test_collected_rollout_is_on_policy(self):
        """Stored log-probs/values equal what the learner's torch policy
        computes on the stored observations: normalization and policy agree."""
        import torch
        from stable_baselines3.common.callbacks import BaseCallback
        from penalty.rl.async_ppo import AsyncCollector
        model, venv = self._model('keeper', 'synthetic')
        venv.obs_rms.mean[:] = .3
        venv.obs_rms.var[:] = 2.
        records = []

        class Recorder(BaseCallback):
            def _on_step(self):
                records.extend(i['episode_record'] for i in self.locals['infos'] if 'episode_record' in i)
                return True

        _, callback = model._setup_learn(10_000, Recorder())
        spec = {'role': 'keeper', 'backend': 'synthetic', 'opponent': 'synthetic', 'backend_options': {}}
        collector = AsyncCollector([dict(spec, seed=s) for s in (1, 2)])
        try:
            self.assertTrue(collector.collect(model, venv, callback, 300))
        finally:
            collector.close()
        buffer = model.rollout_buffer
        self.assertGreaterEqual(buffer.buffer_size, 300)
        self.assertEqual(model.num_timesteps, buffer.buffer_size)
        obs = torch.as_tensor(buffer.observations[:, 0])
        with torch.no_grad():
            values, log_probs, _ = model.policy.evaluate_actions(
                obs, torch.as_tensor(buffer.actions[:, 0]))
        np.testing.assert_allclose(log_probs.numpy(), buffer.log_probs[:, 0], atol=1e-3)
        np.testing.assert_allclose(values.numpy().ravel(), buffer.values[:, 0], atol=1e-4)
        self.assertEqual(buffer.episode_starts[0, 0], 1.)
        self.assertGreater(len(records), 0)
        self.assertGreater(venv.obs_rms.count, 300)
        venv.close()

    def test_async_ppo_learns(self):
        """Synthetic static keeper: random shots score 6.5 %; on 2026-10-09
        20k steps of PPO reached 0.44-0.97 and async PPO 0.83-0.90 (3 seeds)."""
        from penalty.rl.async_ppo import AsyncCollector
        from penalty.rl.evaluation import evaluate
        from penalty.rl.training import save_policy
        model, venv = self._model(opponent='static')
        spec = {'role': 'striker', 'backend': 'synthetic', 'opponent': 'static', 'backend_options': {}}
        model.collector = AsyncCollector([dict(spec, seed=s) for s in (1, 2, 3)])
        try:
            model.learn(20_000)
        finally:
            model.collector.close()
        with tempfile.TemporaryDirectory() as directory:
            save_policy(model, venv, Path(directory), 'striker', {'stage': 'test'})
            result = evaluate('striker', directory, opponent='static', episodes=100, seeds=(7,))
        self.assertGreater(result['summary']['rates']['goal_rate'], .5)
        venv.close()


class RandomizationTests(unittest.TestCase):
    def test_default_shot_keeps_original_draws(self):
        from penalty.rl.bhuman_world import sample_shot
        from penalty.rl.contracts import KICK_LEFT, KICK_RIGHT, TARGET_Y_LIMIT
        a, b = np.random.default_rng(5), np.random.default_rng(5)
        shot = sample_shot(a)
        self.assertEqual((shot.target_y, shot.kick_side, shot.kick_power),
                         (float(b.uniform(-TARGET_Y_LIMIT, TARGET_Y_LIMIT)),
                          int(b.integers(KICK_LEFT, KICK_RIGHT+1)), float(b.uniform(.25, 1.))))

    def test_corner_shots(self):
        from penalty.rl.bhuman_world import CENTER_SHOT_Y, sample_shot
        rng = np.random.default_rng(0)
        shots = [sample_shot(rng, (.4, .65), .25, (.6, 1.)) for _ in range(2000)]
        aims = np.array([abs(s.target_y) for s in shots])
        center = aims <= CENTER_SHOT_Y
        self.assertAlmostEqual(center.mean(), .25, delta=.03)
        self.assertTrue(((aims[~center] >= .4) & (aims[~center] <= .65)).all())
        self.assertAlmostEqual(np.mean([s.target_y > 0 for s in shots]), .5, delta=.03)
        self.assertTrue(all(.6 <= s.kick_power <= 1. for s in shots))

    def test_layout_jitter_bounds_and_no_draws_when_off(self):
        from penalty.rl.bhuman_world import BHumanWorld
        from penalty.rl.geometry import PENALTY_MARK
        rng = np.random.default_rng(1)
        state = rng.bit_generator.state
        self.assertEqual(BHumanWorld('striker', 'scripted')._layout(rng, {}), (PENALTY_MARK, None))
        self.assertEqual(rng.bit_generator.state, state)
        world = BHumanWorld('keeper', 'random_kicks', ball_jitter=.05, start_jitter=(.05, .1))
        for _ in range(200):
            ball, offsets = world._layout(rng, {})
            self.assertLessEqual(abs(ball[0] - PENALTY_MARK[0]), .05)
            self.assertLessEqual(abs(ball[1] - PENALTY_MARK[1]), .05)
            for dx, dy, dyaw in offsets.values():
                self.assertTrue(abs(dx) <= .05 and abs(dy) <= .05 and abs(dyaw) <= .1)
        self.assertEqual(world._layout(rng, {'ball_xy': (.2, .1)})[0], (.2, .1))

    def test_invalid_randomization(self):
        from penalty.rl.bhuman_world import BHumanWorld
        for options in ({'ball_jitter': .5}, {'start_jitter': (.05,)}, {'start_jitter': (.05, 1.)},
                        {'shot_target': (.5, .3)}, {'shot_target': (.3, .9)}, {'center_shots': 1.5},
                        {'shot_power': (0., 2.)}):
            with self.assertRaises(ValueError):
                BHumanWorld('keeper', 'random_kicks', **options)

    def test_synthetic_commit_after_shot(self):
        from penalty.rl.contracts import KeeperAction
        from penalty.rl.evaluation import make_env
        dive = KeeperAction(1, 1., .5).to_normalized()
        legacy = make_env('keeper', 'synthetic', 'synthetic', {'kick_time_range': (2., 2.)})
        legacy.reset(seed=0)
        self.assertEqual(legacy.step(dive)[4]['outcome'], 'keeper_fall')
        gated = make_env('keeper', 'synthetic', 'synthetic',
                         {'kick_time_range': (2., 2.), 'commit_after_shot': True})
        gated.reset(seed=0)
        infos = [gated.step(dive)[4] for _ in range(19)]  # t < 2 s: before the kick
        self.assertTrue(all(i['safety'] == 'waiting_for_shot' and i['outcome'] is None for i in infos))
        while True:
            *_, terminated, truncated, info = gated.step(dive)
            if terminated or truncated:
                break
        self.assertNotEqual(info['outcome'], 'keeper_fall')
        self.assertFalse(info['episode_record']['early_dive'])

    def test_side_summary(self):
        from penalty.rl.metrics import shot_side, side_summary
        self.assertEqual([shot_side(y) for y in (.5, .1, -.5)], ['left', 'center', 'right'])
        records = [{'outcome': 'goal', 'target_y': .65, 'goal_line_y': .55},
                   {'outcome': 'blocked', 'target_y': .6, 'goal_line_y': None},
                   {'result': 'goal', 'target_y': -.65, 'goal_line_y': -.6},
                   {'outcome': 'timeout', 'target_y': None, 'goal_line_y': None}]
        sides = side_summary(records)
        self.assertEqual((sides['kicks'], sides['crossed_goal_line']), (3, 2))
        self.assertAlmostEqual(sides['aim']['left'], 2/3)
        self.assertEqual(sides['goal_line'], {'left': .5, 'center': 0., 'right': .5})
        self.assertEqual(sides['goal_rate_by_aim'], {'left': .5, 'center': None, 'right': 1.})

    def test_striker_records_carry_sides(self):
        from penalty.rl.evaluation import evaluate
        summary = evaluate('striker', 'scripted', opponent='static', episodes=20, seeds=(3,))['summary']
        self.assertEqual(summary['sides']['kicks'], 20)
        self.assertEqual(summary['sides']['aim']['center'], 1.)  # the scripted shot is centred

    def test_perception_noise_option(self):
        from penalty.rl.evaluation import make_env
        from penalty.rl.wrappers import PerceptionNoiseWrapper
        env = make_env('keeper', 'synthetic', 'synthetic', {'perception_noise': {'dropout': .5}})
        self.assertIsInstance(env, PerceptionNoiseWrapper)
        self.assertEqual(env.dropout, .5)
        clean = make_env('keeper', 'synthetic', 'synthetic', {'perception_noise': False})
        self.assertNotIsInstance(clean, PerceptionNoiseWrapper)
        a, _ = env.reset(seed=3)
        b, _ = clean.reset(seed=3)
        self.assertFalse(np.allclose(a, b))


@unittest.skipUnless(HAS_SB3, 'stable-baselines3 not installed')
class ExplorationTests(unittest.TestCase):
    def test_eval_episodes_stay_out_of_rollout_and_std_is_capped(self):
        import torch
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from penalty.rl.async_ppo import AsyncCollector, AsyncPPO
        from penalty.rl.training import build_env
        venv = VecNormalize(DummyVecEnv([lambda: build_env('striker', 'synthetic', 'static')]),
                            norm_reward=False)
        model = AsyncPPO('MlpPolicy', venv, n_steps=600, batch_size=100, device='cpu',
                         policy_kwargs={'net_arch': [128, 128]})
        model.min_std, model.max_std = .2, .5
        with torch.no_grad():
            model.policy.log_std.copy_(torch.log(torch.tensor([.1, 2., .3, 1.])))
        seen = {'eval': 0, 'train': 0}

        class Count(BaseCallback):
            def _on_step(self):
                for info in self.locals['infos']:
                    seen['eval'] += 'eval_record' in info
                    seen['train'] += 'episode_record' in info
                return True

        _, callback = model._setup_learn(10_000, Count())
        spec = {'role': 'striker', 'backend': 'synthetic', 'opponent': 'static', 'backend_options': {},
                'eval_every': 3}
        model.collector = AsyncCollector([dict(spec, seed=s) for s in (1, 2)])
        try:
            self.assertTrue(model.collect_rollouts(venv, callback, None, 600))
        finally:
            model.collector.close()
        np.testing.assert_allclose(model.policy.log_std.exp().detach().numpy(), [.2, .5, .3, .5], rtol=1e-5)
        self.assertGreater(seen['eval'], 0)
        self.assertAlmostEqual(seen['eval']/(seen['eval'] + seen['train']), 1/3, delta=.1)
        buffer = model.rollout_buffer
        self.assertEqual(model.num_timesteps, buffer.buffer_size)
        with torch.no_grad():
            _, log_probs, _ = model.policy.evaluate_actions(torch.as_tensor(buffer.observations[:, 0]),
                                                            torch.as_tensor(buffer.actions[:, 0]))
        np.testing.assert_allclose(log_probs.numpy(), buffer.log_probs[:, 0], atol=1e-3)
        venv.close()

    def test_convergence_uses_eval_episodes_when_configured(self):
        from stable_baselines3 import PPO
        from penalty.rl.training import ConvergenceMonitor, Stage, _Progress, _callback, build_env
        stage = Stage('s', 'striker', 'bhuman', 'static', eval_every=5)
        monitor = ConvergenceMonitor('striker', window=2, patience=9, min_seconds=0)
        with tempfile.TemporaryDirectory() as directory:
            progress = _Progress(Path(directory)/'progress.json')
            callback = _callback(stage, None, progress, lambda: None, monitor)
            model = PPO('MlpPolicy', build_env('striker', 'synthetic', 'empty'), n_steps=64, batch_size=64)
            model._setup_learn(1000, callback)
            callback.on_training_start({}, {})
            for info in ({'episode_record': {'outcome': 'blocked'}}, {'episode_record': {'outcome': 'blocked'}},
                         {'eval_record': {'outcome': 'goal'}}, {'eval_record': {'outcome': 'goal'}}):
                callback.update_locals({'infos': [info]})
                self.assertTrue(callback.on_step())
        self.assertEqual(len(monitor.history), 1)
        self.assertEqual(monitor.best, 1.)

    def test_plan_conditions(self):
        from penalty.rl.policies.baselines import make_policy
        from penalty.rl.training import BASELINES, overnight_plan
        stages = {s.name: s for s in overnight_plan()}
        self.assertTrue(all(s.backend_options.get('perception_noise') for s in stages.values()))
        keeper = stages['keeper_bhuman_random_kicks']
        self.assertEqual(keeper.backend_options['shot_target'], (.3, .65))
        for name in ('keeper_synth_easy', 'keeper_synth_full', 'keeper_bhuman_random_kicks'):
            self.assertTrue(stages[name].backend_options['commit_after_shot'])
        for name in ('keeper_synth_easy', 'keeper_synth_full'):
            self.assertEqual(stages[name].backend_options['kick_time_range'], (3., 9.))
        self.assertEqual((keeper.eval_every, keeper.window), (5, 200))
        self.assertEqual(stages['striker_bhuman_scripted'].backend_options['ball_jitter'], .05)
        self.assertEqual(BASELINES['keeper'], ('geometric', 'center'))
        action = make_policy('center', 'keeper')(np.zeros(19, np.float32))
        np.testing.assert_allclose(action, [0., 1., 0.])

    def test_player_spec(self):
        from penalty.rl.duel import player_spec
        self.assertEqual(player_spec('scripted'), ('scripted', {'kind': 'baseline', 'name': 'scripted'}))
        name, spec = player_spec('v4=models/rl/keeper_v4:gate')
        self.assertEqual((name, spec['kind'], spec['role'], spec['gate'], spec['history']),
                         ('v4', 'trained', 'keeper', True, 6))
        name, spec = player_spec('models/rl/striker_v2')
        self.assertEqual((name, spec['role'], spec['gate']), ('striker_v2', 'striker', False))

    def test_eval_every_one_is_rejected(self):
        from penalty.rl.training import Stage, worker_specs
        with self.assertRaises(ValueError):
            worker_specs(Stage('s', 'striker', 'bhuman', 'static', eval_every=1), 2, 0)


class SupervisorTests(unittest.TestCase):
    def _supervise(self, codes, durations=None, max_restarts=3, progress=True):
        from train_rl import supervise
        calls, now = [], [0.]
        durations = iter(durations or [0.]*len(codes))
        codes = iter(codes)

        def run(resume):
            calls.append(resume)
            now[0] += next(durations)
            return next(codes)

        with tempfile.TemporaryDirectory() as directory:
            if progress:
                (Path(directory)/'progress.json').write_text('{}')
            code = supervise(run, directory, max_restarts, sleep=lambda s: None,
                             clock=lambda: now[0])
        return code, calls

    def test_restarts_crashes_with_resume(self):
        self.assertEqual(self._supervise([1, 75, 0]), (0, [False, True, True]))

    def test_deliberate_abort_is_not_restarted(self):
        self.assertEqual(self._supervise([3]), (3, [False]))

    def test_gives_up_after_repeated_fast_failures(self):
        code, calls = self._supervise([1]*5, max_restarts=2)
        self.assertEqual((code, len(calls)), (1, 3))

    def test_long_healthy_run_resets_failure_count(self):
        code, calls = self._supervise([1, 1, 1, 0], durations=[0, 7200, 0, 0], max_restarts=1)
        self.assertEqual((code, len(calls)), (0, 4))

    def test_failure_before_any_progress_is_not_restarted(self):
        self.assertEqual(self._supervise([1], progress=False), (1, [False]))

    def test_abort_code_matches_training(self):
        from train_rl import EXIT_ABORTED
        if HAS_SB3:
            from penalty.rl.training import EXIT_ABORTED as TRAINING_ABORTED
            self.assertEqual(EXIT_ABORTED, TRAINING_ABORTED)


@unittest.skipUnless(HAS_SB3, 'stable-baselines3 not installed')
class TrainingTests(unittest.TestCase):
    def test_plan_shares_bhuman_hours_per_role(self):
        from penalty.rl.training import overnight_plan
        stages = overnight_plan()
        self.assertEqual(len({s.name for s in stages}), len(stages))
        bhuman = [s for s in stages if s.backend == 'bhuman']
        self.assertAlmostEqual(sum(s.share for s in bhuman), 1.)
        self.assertTrue(all(s.timesteps > 0 for s in stages if s.backend == 'synthetic'))
        self.assertEqual(stages[-1].backend, 'bhuman')

    def test_trained_keeper_round_trip(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from penalty.rl.policies.baselines import make_policy
        from penalty.rl.training import build_env, save_policy
        venv = VecNormalize(DummyVecEnv([lambda: build_env('keeper', 'synthetic', 'synthetic')]),
                            norm_reward=False)
        model = PPO('MlpPolicy', venv, n_steps=128, batch_size=64, n_epochs=1, seed=0,
                    device='cpu')
        model.learn(256)
        with tempfile.TemporaryDirectory() as directory:
            save_policy(model, venv, Path(directory), 'keeper', {'stage': 'test'})
            policy = make_policy(directory, 'keeper')
            self.assertEqual(policy.frames.maxlen, 6)
            action = policy(np.zeros(19, dtype=np.float32))
            self.assertEqual(action.shape, (3,))
            self.assertTrue(np.all(np.abs(action) <= 1))
            with self.assertRaises(ValueError):
                make_policy(directory, 'striker')
            result = evaluate('keeper', directory, episodes=5, seeds=(0,))
            self.assertEqual(result['summary']['episodes'], 5)
        venv.close()


if __name__ == '__main__':
    unittest.main()
