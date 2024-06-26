import numpy as np
import math as math

class AntoineEquationBase10:
    """
    A class that represents the Antoine equation for calculating the saturation pressure of pure components.

    The Antoine equation is a semi-empirical correlation between vapor pressure and temperature for pure components.

    Args:
        A (float): Antoine equation parameter.
        B (float): Antoine equation parameter.
        C (float): Antoine equation parameter.

    Attributes:
        A (float): Antoine equation parameter.
        B (float): Antoine equation parameter.
        C (float): Antoine equation parameter.
    """

    def __init__(self, A: float, B: float, C: float):

        self.A = A
        self.B = B
        self.C = C

    def get_partial_pressure(self, temp:float):
        """
        Calculates the saturation pressure at a given temperature using the Antoine equation.

        Args:
            Temp (np.ndarray): The temperature(s) at which to calculate the saturation pressure.

        Returns:
            np.ndarray: The calculated saturation pressure(s).
        """
        return math.pow(10,(self.A - (self.B/(temp + self.C))))

    def get_temperature(self, partial_pressure: np.ndarray):
        """
        Calculates the temperature at a given saturation pressure using the Antoine equation.

        Args:
            partial_pressure (np.ndarray): The saturation pressure(s) at which to calculate the temperature.

        Returns:
            np.ndarray: The calculated temperature(s).
        """
        # print("boiling point",(self.B/(self.A - np.log10(partial_pressure))) - self.C)
        return (self.B/(self.A - np.log10(partial_pressure))) - self.C
    
    def get_boiling_point(self, P_sys:float) -> float:
        return self.get_temperature(P_sys)
    
    def get_dPsatdT(self, T:float) -> float:
        return (np.log(10)*np.power(10, (self.A*(self.C+T)-self.B)/(self.C+T))*self.B)/(T+self.C)**2

    
class AntoineEquationBaseE:
    """
    A class that represents the Antoine equation for calculating the saturation pressure of pure components.

    The Antoine equation is a semi-empirical correlation between vapor pressure and temperature for pure components.

    Args:
        A (float): Antoine equation parameter.
        B (float): Antoine equation parameter.
        C (float): Antoine equation parameter.

    Attributes:
        A (float): Antoine equation parameter.
        B (float): Antoine equation parameter.
        C (float): Antoine equation parameter.
    """

    def __init__(self, A: float, B: float, C: float):
        self.A = A
        self.B = B
        self.C = C

    def get_partial_pressure(self, temp:float):
        """
        Calculates the saturation pressure at a given temperature using the Antoine equation.

        Args:
            Temp (np.ndarray): The temperature(s) at which to calculate the saturation pressure.

        Returns:
            np.ndarray: The calculated saturation pressure(s).
        """
        return math.exp((self.A - self.B/(temp + self.C)))

    def get_temperature(self, partial_pressure: np.ndarray):
        """
        Calculates the temperature at a given saturation pressure using the Antoine equation.

        Args:
            partial_pressure (np.ndarray): The saturation pressure(s) at which to calculate the temperature.

        Returns:
            np.ndarray: The calculated temperature(s).
        """
        # print("boiling point",(self.B/(self.A - np.log10(partial_pressure))) - self.C)
        return (self.B/(self.A - np.log(partial_pressure))) - self.C
    
    def get_boiling_point(self, P_sys:float) -> float:
        return self.get_temperature(P_sys)

    
