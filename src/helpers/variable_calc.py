class VariableCalculators():
    
    @staticmethod
    def calculate_density(temperature, altitude):
        coefficient = 352.9/(temperature+273)
        brackets = 1-(0.00003379*altitude) #positive and so works for values less than ~30000
        density = coefficient * (brackets**3.5075)
        return str(round(density, 5))

    @staticmethod
    def calculate_temperature(density):
        if density > 0.75:
            altitude = 0
        else:
            altitude = 12200
            
        temperature_kelvin = (101325 - 9.8*altitude*density)/(287 * density)
        temperature = temperature_kelvin - 273
        return str(round(temperature, 5))
    


# need to add during testing due to calculated values sometimes being out?
"""
        if brackets < 0.1: #adds stability to the calculated value
            brackets = 0.1

"""