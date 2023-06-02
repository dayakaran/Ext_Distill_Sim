class ThermodynamicModel:
    def convert_x_to_y(self, x_i):
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction 

        Args:
            x (int or float): liquid mole fraction of a component

        Raises:
            NotImplementedError: base class
        """
        raise NotImplementedError("Method convert_x_to_y not implemented in base class")

    def convert_y_to_x(self, y_i):
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction 

        Args:
            y (int or float): vapor mole fraction of a component

        Raises:
            NotImplementedError: base class
        """
        raise NotImplementedError("Method convert_y_to_x not implemented in base class")

class RaoultsLawModel(ThermodynamicModel):
    def __init__(self, Pressure_system):
        self.P_sys = Pressure_system
    
    def convert_x_to_y(self, *x_i, P_isat):
        
        