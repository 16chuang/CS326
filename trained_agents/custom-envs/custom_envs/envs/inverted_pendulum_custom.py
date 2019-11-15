import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class InvertedPendulumCustomEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 gravity=-9.81,
                 pole_radius=0.049,
                 pole_half_len=0.3,
                 pole_mass=1,
                 sliding_friction=1):
        utils.EzPickle.__init__(self)

        xml_filepath = '{}/assets/InvertedPendulum-v2/inverted_pendulum_g{}_r{}_l{}_m{}_f{}.xml'.format(
            os.path.dirname(__file__),
            str(gravity).replace('-', '').replace('.', ''),
            str(pole_radius).replace('.', ''),
            str(pole_half_len).replace('.', ''),
            str(pole_mass).replace('.', ''),
            str(sliding_friction).replace('.', ''))

        if not os.path.exists(xml_filepath):
            xml_str = '''<mujoco model="inverted pendulum">
                <compiler inertiafromgeom="true"/>
                <default>
                    <joint armature="0" damping="1" limited="true"/>
                    <geom contype="0" friction="{} 0.1 0.1" rgba="0.7 0.7 0 1"/>
                    <tendon/>
                    <motor ctrlrange="-3 3"/>
                </default>
                <option gravity="0 0 {}" integrator="RK4" timestep="0.02"/>
                <size nstack="3000"/>
                <worldbody>
                    <!--geom name="ground" type="plane" pos="0 0 0" /-->
                    <geom name="rail" pos="0 0 0" quat="0.707 0 0.707 0" rgba="0.3 0.3 0.7 1" size="0.02 1" type="capsule"/>
                    <body name="cart" pos="0 0 0">
                        <joint axis="1 0 0" limited="true" name="slider" pos="0 0 0" range="-1 1" type="slide"/>
                        <geom name="cart" pos="0 0 0" quat="0.707 0 0.707 0" size="0.1 0.1" type="capsule"/>
                        <body name="pole" pos="0 0 0">
                            <joint axis="0 1 0" name="hinge" pos="0 0 0" range="-90 90" type="hinge"/>
                            <geom fromto="0 0 0 0.001 0 0.6" name="cpole" rgba="0 0.7 0.7 1" size="{} {}" type="capsule" mass="{}"/>
                        </body>
                    </body>
                </worldbody>
                <actuator>
                    <motor gear="100" joint="slider" name="slide"/>
                </actuator>
            </mujoco>'''.format(sliding_friction, gravity, pole_radius,
                                pole_half_len, pole_mass)

            with open(xml_filepath, 'w') as file:
                file.write(xml_str)

        mujoco_env.MujocoEnv.__init__(self, xml_filepath, 2)

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent


class InvertedPendulumGravity1Env(InvertedPendulumCustomEnv):
    def __init__(self):
        super().__init__(gravity=-1.0)


class InvertedPendulumGravity5Env(InvertedPendulumCustomEnv):
    def __init__(self):
        super().__init__(gravity=-5.0)


class InvertedPendulumGravity20Env(InvertedPendulumCustomEnv):
    def __init__(self):
        super().__init__(gravity=-20.0)


class InvertedPendulumGravity50Env(InvertedPendulumCustomEnv):
    def __init__(self):
        super().__init__(gravity=-50.0)
