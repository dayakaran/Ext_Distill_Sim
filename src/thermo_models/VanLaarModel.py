import numpy as np
from utils.AntoineEquation import *
from thermo_models import VLEModel

class VanLaarModel(VLEModel):
    """
    Implements the Van Laar model for estimating activity coefficients in binary or multicomponent systems.

    Parameters:
        num_comp (int): Number of components in the mixture.
        P_sys (float): System pressure, assumed to be constant, in units compatible with the Antoine equation.
        comp_names (list): Names of the components in the mixture.
        partial_pressure_eqs (AntoineEquationBase10): Antoine equation parameters for calculating vapor pressures.
        A_coeff (dict): Van Laar A coefficients, where keys are tuples of component indices (i, j) and values are the coefficients.
        use_jacobian (bool, optional): Indicates whether to use the Jacobian matrix for numerical methods. Defaults to False.

    Attributes:
        P_sys (float): The system pressure used in calculations.
        A_coeff (dict): Van Laar A coefficients for the mixture components.
    """

    
    def __init__(self, num_comp: int, P_sys: float, comp_names, partial_pressure_eqs: AntoineEquationBase10, A_coeff: dict, use_jacobian=False):
        super().__init__(num_comp, P_sys, comp_names, partial_pressure_eqs, use_jacobian)
        self.A_coeff = A_coeff

    def get_activity_coefficient(self, x_array, Temp:float):
        """
        Calculates activity coefficients using the Van Laar equation for each component in the mixture.

        This method determines the deviation from ideal solution behavior by calculating the activity coefficients
        based on the Van Laar model, which requires the mixture's composition, temperature, and Van Laar A coefficients.

        Parameters:
            x_array (np.ndarray): Mole fractions of components in the liquid phase.
            Temp (float): Temperature at which to calculate the activity coefficients.

        Returns:
            list: Activity coefficients for each component in the mixture.

        Raises:
            ValueError: If A coefficients are entered incorrectly or if any mathematical operation results in an invalid value.
        """

        #Assert that A_coeff[(i,i)] = 1
        for i in range(1, self.num_comp+1):
            if (self.A_coeff[(i,i)] != 0):
                raise ValueError('A Coefficients entered incorrectly')
            
        gammas = []
        z_array = [] # First compute the mole fractions 
        for i in range(1, self.num_comp+1):
            denom = 0
            for j in range(1, self.num_comp+1): #summation over j term in denominator
                if (i == j): # Aii = 0
                    denom += (x_array[j-1])  # "If Aji / Aij = 0 / 0, set Aji / Aij = 1" -- Knapp Thesis
                else:
                    denom += (x_array[j-1] * self.A_coeff[(j,i)] / self.A_coeff[(i,j)])
            z_array.append((x_array[i - 1] / denom))
        
        for k in range(1, self.num_comp+1):
            term1 = 0 # summation of Aki * zi
            term2 = 0 # summation of Aki * zk * zi
            term3 = 0 # The sum of sum of Aji * (Akj / Ajk) * zj * zi
            for i in range(1, self.num_comp+1):
                term1 += (self.A_coeff[(k,i)] * z_array[i-1])
                term2 += (self.A_coeff[(k,i)] * z_array[k-1] * z_array[i-1]) 
            for j in range(1, self.num_comp+1):
                for i in range(1, self.num_comp+1):
                    if (j == k or i == k): # "If Aji / Aij = 0 / 0, set Aji / Aij = 1" -- Knapp
                        pass
                    else:                      
                        term3 += (self.A_coeff[(j,i)] * self.A_coeff[(k,j)] / self.A_coeff[(j,k)] * z_array[j-1] * z_array[i-1])
            gammas.append(math.exp((term1 - term2 - term3)/Temp))
        return gammas

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
