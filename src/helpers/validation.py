import re
#these apparently dont work :(
density_validation = re.compile(r"^(0?\.(?!0+)\d{1,5}) | (1(\.\d{0,5})?) | (2(\.0{0,5})?)$")
temperature_validation = re.compile(r"^-?[012345]\d?(\.\d{0,5})?$")
altitude_validation = re.compile(r"^([012]\d{0,4}) | (\d{1,4}) (\.\d{0,5})?$")
sim_width_validation = re.compile(r"^(0?\.(?!0+)\d{0,5}) | ([12]?\d(\.\d{0,5})?) | (3\d(\.\d{0,5})?)$")
inflow_velocity_validation = re.compile(r"^(0?\.(?!0+)\d{0,5}) | (1?\d{1,2}(\.\d{0,5})?) | (2[01234]\d(\.\d{0,5})?)$")

class Validators(object):

    def __init__(self, density_val, temperature_val, altitude_val, sim_width_val, inflow_velocity_val):
        self.density_regex = density_val
        self.temperature_regex = temperature_val
        self.altitude_regex = altitude_val
        self.sim_width_regex = sim_width_val
        self.inflow_velocity_regex = inflow_velocity_val
    
    def validate_density(self, input):
        if self.density_regex.match(input):
            return True
        else:
            return False
    
    def validate_temperature(self, input):
        if self.temperature_regex.match(input):
            return True
        else:
            return False

    def validate_altitude(self, input):
        if self.altitude_regex.match(input):
            return True
        else:
            return False

    def validate_sim_width(self, input):
        if self.sim_width_regex.match(input):
            return True
        else:
            return False

    def validate_inflow_velocity(self, input):
        if self.inflow_velocity_regex.match(input):
            return True
        else:
            return False

validators = Validators(density_validation, temperature_validation, altitude_validation, sim_width_validation, inflow_velocity_validation)

validation_dict = {
    "density": validators.validate_density,
    "temperature": validators.validate_temperature,
    "altitude": validators.validate_altitude,
    "sim_width": validators.validate_sim_width,
    "inflow_velocity": validators.validate_inflow_velocity,

}