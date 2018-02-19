import gym
import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class AsteroidEnv(gym.Env):
    """
    An environment where the actor must shoot a stationary asteroid,
    the position of the asteroid is relative to the actor,
    the actor only gets one shot
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.projectile_radius = 0.1
        self.rotation_speed = 5
        self.movement_speed = 10
        self.asteroid_position = np.asarray([0.0,0.0])
        self.sim_time_mult = 0.1
        self.asteroid_radius = 1

        # move left/right, move forwards/back, rotate, shoot/not-shoot
        self.action_space = spaces.MultiDiscrete([2,2,2,2])
        observed_maxes = np.array([np.finfo(np.float32).max,np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-observed_maxes,observed_maxes)

        self.seed()

        # initialized in reset()
        self.state = None
        self.actor_position = None
        self.actor_rotation = None
        self.step = 0

        self.viewer = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        # start actor in random location in a 40X40 box around the origin with a random rotation
        self.actor_position = self.np_random.uniform(low=-20, high=20, size=(2,))
        self.actor_rotation = self.np_random.uniform(low=0, high=2 * math.pi)

        self.step = 0

        # the state stores information relative to actor
        self.state = self.world_to_local_space(self.actor_position, self.actor_rotation, self.asteroid_position)
        self.steps_beyond_done = None
        return np.array(self.state)

    def world_to_local_space(self, space_position, space_rotation, input_position):
        """
        convert a position from world to local space
        :param space_position:        position of the local space in world coordinates, a (2,) numpy array
        :param space_rotation:        rotation of the local space in world space, a float [0,2pi)
        :param input_position:  position to convert to local space
        :return:                the "input_position" in local space
        """
        new_x = -space_position[0] + (input_position[0] *  math.cos(space_rotation) + input_position[1] * math.sin(space_rotation))
        new_y = -space_position[1] + (input_position[0] * -math.sin(space_rotation) + input_position[1] * math.cos(space_rotation))
        return np.asarray([new_x,new_y])

    def local_to_world_space(self,space_position, space_rotation, input_position):
        """convert a postion from local space to world space"""
        new_x = space_position[0] + (input_position[0] * math.cos(space_rotation) + input_position[1] * -math.sin(space_rotation))
        new_y = space_position[1] + (input_position[0] * math.sin(space_rotation) + input_position[1] *  math.cos(space_rotation))
        return np.asarray([new_x,new_y])

    def step(self,action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        self.step += 1

        # handle rotation input
        self.actor_rotation += (float(action[2]) - 0.5) * 2 * self.rotation_speed * self.sim_time_mult

        # handle movement input in local space
        local_movement = np.asarray([0.0,0.0])
        local_movement[1] += (float(action[0]) - 0.5)* 2 * self.movement_speed * self.sim_time_mult # left/right
        local_movement[0] += (float(action[1]) - 0.5)* 2 * self.movement_speed * self.sim_time_mult # forwards/back

        # convert to world movement and apply to position
        self.actor_position = self.local_to_world_space(self.actor_position, self.actor_rotation, local_movement)

        state = self.world_to_local_space(self.actor_position, self.actor_rotation, self.asteroid_position)

        reward = 0.0
        done = False

        # check if the target has been shot
        if action[3] == 1: # shot fired
            done = True
            # target is hit if it is along the local x axis (state holds asteroid position in local space)
            if abs(state[0]) < self.asteroid_radius:
                # reward is higher if task is done faster
                reward = max((100.0 - self.step*.001),10)

        self.state = state

        return np.array(self.state),reward,done,{}

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        view_window_width = 50
        scale = screen_width/view_window_width
        size_of_asteroid = self.asteroid_radius
        actor_width = self.asteroid_radius

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width,screen_height)
            l,r,t,b = (-actor_width/2)*scale,(actor_width/2)*scale,(actor_width)*scale,-(actor_width)*scale
            actor = rendering.FilledPolygon([(l,b),[0,t],(r,b)])# triangle pointing in the x axis
            self.actor_trans = rendering.Transform()
            actor.add_attr(self.actor_trans)
            self.viewer.add_geom(actor)
            asteroid = rendering.make_circle(size_of_asteroid*scale)
            self.asteroid_trans = rendering.Transform()
            asteroid.set_color(.8,.6,.4)
            self.asteroid_trans = rendering.Transform()
            asteroid.add_attr(self.asteroid_trans)
            self.viewer.add_geom(asteroid)

        if self.state is None: return None

        asteroid_x = (self.asteroid_position[0] * scale) + screen_width /2.0
        asteroid_y = (self.asteroid_position[1] * scale) + screen_width /2.0

        self.asteroid_trans.set_translation(asteroid_x,asteroid_y)

        actor_x = (self.actor_position[0] * scale) + screen_width / 2.0
        actor_y = (self.actor_position[1] * scale) + screen_height / 2.0

        print(self.actor_position)

        self.actor_trans.set_translation(actor_x, actor_y)
        self.actor_trans.set_rotation(self.actor_rotation)

        return self.viewer.render(return_rgb_array = (mode == 'rbg_array'))

