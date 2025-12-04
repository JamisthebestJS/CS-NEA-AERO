#may not end up using this file, but reminder to implement anyway

import re

#density
denisty_validation = "(?:0?\.(?!0+$)\d{1,5}) | (1(?:\.\d{1,5})?) | (2(?:\.0{1,5})?)"
re.search(denisty_validation)

class Validators(object):

    def __init__(self, density, temperature, altitude, sim_scale, inflow_velocity):
        self.density_regex = density
        self.temperature_regex = temperature
        self.altitude_regex = altitude
        self.sim_scale_regex = sim_scale
        self.inflow_velocity_regex = inflow_velocity
    
    def validate_density(self, input):
        re.search()
        pass