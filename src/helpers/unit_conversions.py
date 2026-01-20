SIM_HEIGHT = 300
SIM_VELOCITY = 0.04
TIME_TO_END_SIM = 1000 #from testing - around 1000 iterations for velocity to propogate to end of simulation space.


class Conversions(object):

    def __init__(self, SI_velocity, SI_density, average_sim_density, SI_length):
        self.velocity_coefficient = self.vel_coef_calc(SI_velocity)
        self.length_coefficient = self.length_coef_calc(SI_length)
        self.density_coefficient = self.density_coef_calc(SI_density, average_sim_density)
        self.force_coefficient = self.force_coef_calc(SI_density, SI_velocity, average_sim_density)


    def SI_to_sim_length(self, SI_length):
        sim_length = SI_length / self.length_coefficient
        return sim_length
    
    def SI_to_sim_density(self, SI_density):
        sim_density = SI_density / self.density_coefficient
        return sim_density
    
    def SI_to_sim_velocity(self, SI_velocity):
        sim_velocity = SI_velocity / self.velocity_coefficient
        return sim_velocity
    
    def sim_to_SI_force(self, sim_force):
        SI_force = sim_force * self.force_coefficient
        return SI_force
    

    def vel_coef_calc(self, SI_velocity):
        return SI_velocity / SIM_VELOCITY
    
    def length_coef_calc(self, SI_length):
        return SI_length / (SIM_HEIGHT*2)
    
    def density_coef_calc(self, SI_density, sim_density):
        return SI_density / sim_density
    
    def force_coef_calc(self, SI_density, SI_velocity, sim_density):
        force_coef = ((SI_velocity**2) * SI_density * (SIM_HEIGHT**2) * (TIME_TO_END_SIM ** 2))/ \
            ((4 * 300 ** 4) * sim_density)
        return force_coef
    
