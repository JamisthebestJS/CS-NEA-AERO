#may not end up using this file, but reminder to implement anyway

import re
#these apparently dont work :(
density_validation = r"^0?\.(?!0+$)\d{1,5})|(1(\.\d{1,5})?)|(2(\.0{1,5})?$"
temperature_validation = r"-?[0-5]\d?(\.\d{1,5})?$"
altitude_validation = r"(0|1|2\d{0,4})|(\d{1,4}) (\.\d{1,5})?$"
sim_scale_validation = r"(0?\.(?!0+$)\d{0,5})|([1-2]?\d(\.\d{1,5})?)|(3\d(\.\d{1,5})?)$"
inflow_velocity_validation = r"(0?\.(?!0+$)\d{1,5})|(1?\d{1,2}(\.\d{1,5})?)|(2[0-4]\d(\.\d{1,5})?)$"



class Validators(object):

    def __init__(self, density, temperature, altitude, sim_scale, inflow_velocity):
        self.density_regex = density
        self.temperature_regex = temperature
        self.altitude_regex = altitude
        self.sim_scale_regex = sim_scale
        self.inflow_velocity_regex = inflow_velocity
    
    def validate_density(self, input):
        found = re.search(self.density_regex, input)
        return found is not None
    
    def validate_temperature(self, input):
        found = re.search(self.temperature_regex, input)
        return found is not None

    def validate_altitude(self, input):
        found = re.search(self.altitude_regex, input)
        return found is not None

    def validate_sim_scale(self, input):
        found = re.search(self.sim_scale_regex, input)
        return found is not None

    def validate_inflow_velocity(self, input):
        found = re.search(self.inflow_velocity_regex, input)
        return found is not None

validators = Validators(density_validation, temperature_validation, altitude_validation, sim_scale_validation, inflow_velocity_validation)

validation_dict = {
    "density": validators.validate_density,
    "temperature": validators.validate_temperature,
    "altitude": validators.validate_altitude,
    "sim_scale": validators.validate_sim_scale,
    "inflow_velocity": validators.validate_inflow_velocity,

}