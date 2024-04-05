import numpy as np
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 

from utils.AntoineEquation import *
from thermo_models.VLEModelBaseClass  import *

class RaoultsLawModel(VLEModel):
    """
    Implements Raoult's Law for vapor-liquid equilibrium in systems with ideal gas and ideal liquid phases.

    Attributes:
        num_comp (int): Number of components in the system.
        P_sys (float): Total system pressure.
        comp_names (list): Names of the components in the system.
        partial_pressure_eqs (list of AntoineEquation): Antoine equations to calculate vapor pressures.
        use_jacobian (bool): Indicates whether to utilize the Jacobian matrix for calculations. Defaults to False.
    """

    
    def __init__(self, num_comp: int, P_sys: float, comp_names, partial_pressure_eqs, use_jacobian=False):
        """
        Initializes a RaoultsLawModel instance with essential parameters for VLE calculations.

        Parameters:
            num_comp (int): Number of components in the mixture.
            P_sys (float): Total system pressure.
            comp_names (list): Names of the components in the mixture.
            partial_pressure_eqs (list of AntoineEquation): Antoine equation parameters for each component.
            use_jacobian (bool, optional): Flag indicating the use of the Jacobian matrix in optimizations. Defaults to False.
        """
        super().__init__(num_comp, P_sys, comp_names, partial_pressure_eqs, use_jacobian)

        
    def compute_gas_partial_fugacity(self,y_i:np.ndarray) -> np.ndarray:
        """
        Calculates the partial fugacity of each gas phase component in a mixture following Raoult's Law.

        Parameters:
            y_i (np.ndarray): Mole fractions of the components in the gas phase.

        Returns:
            np.ndarray: Partial fugacities of the components in the gas phase, assuming ideal behavior.
        """
        return y_i * self.P_sys
    
    def get_vapor_pressure(self, Temp)->np.ndarray:
        """
        Computes the vapor pressure for each component at a given temperature.
        
        Args:
            Temp (float): The temperature at which to compute the vapor pressure.
            
        Returns:
            np.ndarray: The vapor pressure for each component.
        """
        vap_pressure_array = []
        for partial_pressure_eq in self.partial_pressure_eqs:
            vap_pressure_array.append(partial_pressure_eq.get_partial_pressure(Temp))
        return np.array(vap_pressure_array)
    
    def get_activity_coefficient(self, x_array, Temp = None):
        """
        Computes the activity coefficient for each component. 
        For a system following Raoult's Law, the activity coefficient is 1.
        
        Returns:
            np.ndarray: The activity coefficient for each component.
        """
        return np.ones(self.num_comp)


        

    
        

