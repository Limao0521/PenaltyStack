"""Explicit simulation adapter for the keeper_motion_skill_v1 contract.

The legacy control omits the extension. A profile only enters keeper block
requests; original standing, walking, kicking, preparation and recovery stay
with B-Human. Profiles can change between complete world resets only.
"""
from dataclasses import asdict
import math

import numpy as np

from penalty.keeper_defenses import DEFENSE_NAMES
from penalty.physical_sim import PhysicalSimulator
from penalty.rl.defense_benchmark import SensorDefenseDuelWorld
from penalty.sim_sensors import load_sensor_model
from penalty.skills.keeper_motion import (MOTION_SKILL_VERSION,
                                         ParameterizedKeeperMotion,
                                         motion_parameters as parse_motion_parameters)


class MotionPhysicalSimulator(PhysicalSimulator):
    def __init__(self, motion_parameters=None, **options):
        self.motion_parameters = parse_motion_parameters(motion_parameters)
        self.motion_evidence = None
        self.max_keeper_joint_velocity = 0.
        self.profile_request_count = 0
        super().__init__(**options)

    def set_profile(self, parameters):
        if self.bridges:
            raise RuntimeError('cambia el perfil sólo antes de reiniciar los runtimes')
        self.motion_parameters = parse_motion_parameters(parameters)

    def reset(self, trial):
        self.motion_evidence = None
        self.max_keeper_joint_velocity = 0.
        self.profile_request_count = 0
        return super().reset(trial)

    def step(self, requests):
        # Copy before injecting: do not modify a caller's shared commands.
        requests = {role: dict(request) for role, request in requests.items()}
        request = requests.get('keeper', {})
        if (self.motion_parameters is not None and request.get('motion') == 'block'
                and request.get('block_side', 'center') in DEFENSE_NAMES):
            request.update(ParameterizedKeeperMotion().apply(self.motion_parameters))
            self.profile_request_count += 1
        observation = super().step(requests)
        output = self.outputs.get('keeper', {})
        profile = output.get('keeper_motion_skill')
        if profile is not None:
            if self.motion_parameters is None:
                raise RuntimeError('el control legacy recibió un perfil de movimiento inesperado')
            wanted = asdict(self.motion_parameters)
            confirmed = profile.get('parameters', {})
            matching = (set(confirmed) == set(wanted)
                        and all(isinstance(confirmed[name], (int, float))
                                and math.isclose(confirmed[name], wanted[name],
                                                 abs_tol=1e-7, rel_tol=1e-7)
                                for name in wanted))
            if (profile.get('version') != MOTION_SKILL_VERSION or not matching
                    or profile.get('applied') is not True):
                raise RuntimeError('el runtime no confirmó el perfil solicitado')
            if self.motion_evidence is None or 'blocks' in profile:
                self.motion_evidence = dict(profile)
        velocity = self.adapter.frame('keeper')['velocity']
        self.max_keeper_joint_velocity = max(self.max_keeper_joint_velocity,
                                            float(np.max(np.abs(velocity))))
        return observation


class MotionDuelWorld(SensorDefenseDuelWorld):
    def __init__(self, motion_parameters=None, **options):
        self.next_motion_parameters = parse_motion_parameters(motion_parameters)
        super().__init__(**options)
        self.simulator_factory = lambda: MotionPhysicalSimulator(
            self.next_motion_parameters, model_factory=load_sensor_model)

    def set_profile(self, parameters):
        """Set the profile for the next reset, leaving an active trial intact."""
        self.next_motion_parameters = parse_motion_parameters(parameters)

    def reset(self, rng, options=None):
        session = self._ensure_session()
        session.sim.close()
        session.sim.set_profile(self.next_motion_parameters)
        return super().reset(rng, dict(options or {}))

    def record(self):
        record = dict(super().record())
        if self.session is not None:
            sim = self.sim
            record.update(motion_skill=(None if sim.motion_parameters is None else
                                        {'version': MOTION_SKILL_VERSION,
                                         'parameters': asdict(sim.motion_parameters)}),
                          motion_evidence=sim.motion_evidence,
                          profile_request_count=sim.profile_request_count,
                          max_keeper_joint_velocity_rad_s=sim.max_keeper_joint_velocity)
            if sim.motion_parameters is not None and record.get('commit_defense') is not None:
                if sim.motion_evidence is None:
                    raise RuntimeError('compromiso sin confirmación del perfil articular')
        return record
