import numpy as np
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir))

sys.path.append(PROJECT_ROOT) 
from utils.AntoineEquation import *
from thermo_models.VLEModelBaseClass  import *

class MargulesModel(VLEModel):

    '''
    Implements the Margules model for estimating activity coefficients in binary or multicomponent systems.

    The Margules model provides a way to calculate activity coefficients based on the system's composition,
    temperature, and specific interaction parameters between components. This class enables the computation
    of vapor pressures and activity coefficients essential for understanding the VLE behavior of the system.

    Parameters:
        num_comp (int): Number of components in the mixture.
        P_sys (float): System pressure, assumed constant, in units compatible with the vapor pressures.
        A_ (dict): Interaction parameters for the Margules model. Dictionary keys are tuples (i, j)
                representing component pairs, with corresponding coefficients as dictionary values.
        comp_names (list): Names of the components in the system.
        partial_pressure_eqs (AntoineEquationBase10): Antoine equation parameters for calculating vapor pressures.
        use_jacob (bool, optional): Indicates whether to use the Jacobian matrix for optimizations. Defaults to False.
    '''


    def __init__(self, num_comp:int, P_sys:np.ndarray, A_:dict, comp_names, partial_pressure_eqs: AntoineEquationBase10, use_jacob:bool):
        '''
        Initializes a MargulesModel instance with essential parameters for VLE calculations.

        Parameters:
            num_comp (int): Number of components in the mixture.
            P_sys (float): Total system pressure.
            A_ (dict): Margules coefficients, where keys are tuples of component indices (i, j) and values are the coefficients.
            comp_names (list): Names of the components in the mixture.
            partial_pressure_eqs (AntoineEquationBase10): Antoine equation parameters for each component.
            use_jacob (bool, optional): Flag indicating the use of the Jacobian matrix in optimizations. Defaults to False.
        '''

        super().__init__(num_comp, P_sys, comp_names,partial_pressure_eqs,use_jacob)
        self.A_ = A_
        
    def get_activity_coefficient(self, x_array:np.ndarray, Temp:float):
        '''
        Calculates activity coefficients for each component in a mixture using the Margules equation.

        For a binary system, specific formulas are used. For multicomponent systems, a generalized form of the
        Margules equation is applied.

        Parameters:
            x_array (np.ndarray): Mole fractions of components in the liquid phase.
            Temp (float): Temperature at which the activity coefficients are calculated.

        Returns:
            np.ndarray: An array of activity coefficients for each component in the mixture.

        Raises:
            ValueError: If the interaction coefficients (A) are entered incorrectly.
        '''

        if (self.num_comp == 2):
            gamma1 = np.exp((self.A_[(1,2)] + 2*(self.A_[(2,1)] - self.A_[(1,2)])*x_array[0]) * (x_array[1]**2))
            gamma2 = np.exp((self.A_[(2,1)] + 2*(self.A_[(1,2)] - self.A_[(2,1)])*x_array[1]) * (x_array[0]**2))     
            return np.array([gamma1, gamma2])
        else:
            A_ = self.A_
            gammas = []

            part3 = 0
            for i in range(self.num_comp):
                for j in range(self.num_comp):
                    part3 += 2.0*A_[i,j]*x_array[i]*(x_array[j]**2)
            
            for k in range(self.num_comp):

                part1 = np.sum(np.array( [A_[k, i] * x_array[i]**2 for i in range(self.num_comp) ]))
                part2 = np.sum(np.array( [A_[i, k] * x_array[i]*x_array[k] for i in range(self.num_comp) ]))
                
                result = np.exp((part1 + part2 - part3)/Temp)
                gammas.append(result)
                
            return np.array(gammas)

    def get_vapor_pressure(self, Temp)->np.ndarray:
        '''
        Args:
            Temp (float): The temperature at which to compute the vapor pressure.      
        Returns:
            np.ndarray: The vapor pressure for each component.
        '''
        vap_pressure_array = []
        for partial_pressure_eq in self.partial_pressure_eqs:
            vap_pressure_array.append(partial_pressure_eq.get_partial_pressure(Temp))
        return np.array(vap_pressure_array)
