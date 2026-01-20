import re
density_validation = r"^(0?\.(?!0+$)\d{1,5}|1(\.\d{1,5})?|2(\.0{1,5})?)$"
temperature_validation = r"^-?[0-5]\d?(\.\d{1,5})?$"
altitude_validation = r"^(1|2)?\d{0,4}(\.\d{1,5})?$"
sim_width_validation = r"^(0?\.(?!0+$)\d{0,5}|(1|2)?\d(\.\d{1,5})?|3\d(\.\d{1,5})?)$"
inflow_velocity_validation = r"^(0?\.(?!0+$)\d{1,5}|1?\d{1,2}(\.\d{1,5})?|2[0-4]\d(\.\d{1,5})?)$"



class Validators(object):

    def __init__(self, density_val, temperature_val, altitude_val, sim_width_val, inflow_velocity_val):
        self.density_regex = density_val
        self.temperature_regex = temperature_val
        self.altitude_regex = altitude_val
        self.sim_width_regex = sim_width_val
        self.inflow_velocity_regex = inflow_velocity_val
    
    def validate_density(self, input):
        search = re.fullmatch(self.density_regex, input)
        return search is not None and input != ""
    
    def validate_temperature(self, input):
        search = re.fullmatch(self.temperature_regex, input)
        return search is not None and input != ""

    def validate_altitude(self, input):
        search = re.fullmatch(self.altitude_regex, input)
        return search is not None and input != ""

    def validate_sim_width(self, input):
        search = re.fullmatch(self.sim_width_regex, input)
        try:
            if int(input) == 0:
                return False
        except:
            return False
        return search is not None and input != ""

    def validate_inflow_velocity(self, input):
        search = re.fullmatch(self.inflow_velocity_regex, input)
        try:
            if int(input) == 0:
                return False
        except:
            return False
        return search is not None and input is not ""

validators = Validators(density_validation, temperature_validation, altitude_validation, sim_width_validation, inflow_velocity_validation)

validation_dict = {
    "density": validators.validate_density,
    "temperature": validators.validate_temperature,
    "altitude": validators.validate_altitude,
    "sim_width": validators.validate_sim_width,
    "inflow_velocity": validators.validate_inflow_velocity,

}