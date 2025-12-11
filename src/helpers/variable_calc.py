class VariableCalculators():
    
    @staticmethod
    def calculate_density(temperature, altitude):
        coefficient = 353/temperature
        brackets = 1-(3.379*10**-5)*altitude
        density = coefficient * (brackets**3.5075)
        return density

    @staticmethod
    def calculate_temperature(density):
        altitude = 12200

        coefficient = 353/density
        brackets = 1-(3.379*10**-5)*altitude
        temperature = coefficient * (brackets**3.5075)
        return temperature