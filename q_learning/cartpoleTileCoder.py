import tiles3 as tc
import numpy as np
class CartpoleTileCoder:
    def __init__(self, iht_size=4096, num_tilings=32, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same

        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """

        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht = tc.IHT(iht_size)

    def get_tiles(self, obs):
        [x, x_dot, costheta, sintheta, ang_vel] = obs
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.

        returns:
        tiles -- np.array, active tiles

        """

        ### Set the max and min of angle and ang_vel to scale the input
        x_threshold=0.35
        theta_dot_threshold=50
        X_MIN = - x_threshold
        X_MAX = x_threshold
        X_MIN_VEL = - x_threshold*20
        X_MAX_VEL = x_threshold*20
        THETA_MIN_VEL = - theta_dot_threshold
        THETA_MAX_VEL = theta_dot_threshold

        ### Use the ranges above and self.num_tiles to set angle_scale and ang_vel_scale (2 lines)
        # angle_scale = number of tiles / angle range
        # ang_vel_scale = number of tiles / ang_vel range

        ### START CODE HERE ###
        x_scale = self.num_tiles / (X_MAX - X_MIN)
        x_vel_scale = self.num_tiles / (X_MAX_VEL - X_MIN_VEL)
        cos_scale = self.num_tiles / 2
        ang_vel_scale = self.num_tiles / (THETA_MAX_VEL - THETA_MIN_VEL)
        ### END CODE HERE ###
        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        # tiles = tc.tileswrap(self.iht, wrapwidths=[self.num_tiles, False], numtilings=self.num_tilings,
        #                      floats=[angle * angle_scale, ang_vel * ang_vel_scale])
        tiles = tc.tileswrap(self.iht, wrapwidths=[self.num_tiles, False], numtilings=self.num_tilings,
                             floats=[x * x_scale, x_dot * x_vel_scale, costheta*cos_scale, sintheta*cos_scale, ang_vel * ang_vel_scale])

        return np.array(tiles)
