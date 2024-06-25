import numpy as np
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir))

sys.path.append(PROJECT_ROOT) 
from thermo_models.VLEModelBaseClass import *

class WilsonModel(VLEModel):
    '''
    A thermodynamic model implementing the Wilson equation for calculating vapor-liquid equilibrium (VLE).

    Parameters:
        num_comp (int): The number of components in the system.
        P_sys (float): The system pressure, assumed constant, in units compatible with the vapor pressures.
        comp_names (list): Names of the components in the system.
        Lambdas (dict): Interaction parameters for the Wilson model. Dictionary keys are tuples (i, j)
                        representing component pairs, with corresponding lambda values as dictionary values.
        partial_pressure_eqs (list, optional): List containing AntoineEquation objects for calculating 
                        the vapor pressure of each component. Default is None.
        use_jacob (bool, optional): Flag indicating whether to use the Jacobian matrix for optimizations. 
                        Default is False.

    Reference:
        Orye, R. V., & Prausnitz, J. M. (1965). MULTICOMPONENT EQUILIBRIA—THE WILSON EQUATION.
        Industrial & Engineering Chemistry, 57(5), 18-26. https://doi.org/10.1021/ie50665a005
    '''

    def __init__(self, num_comp: int, P_sys: float, comp_names, Lambdas: dict, partial_pressure_eqs=None, use_jacob=False):

        super().__init__(num_comp, P_sys, comp_names, partial_pressure_eqs, use_jacob)
        self.Lambdas = Lambdas

    def get_activity_coefficient(self, x_, Temp = None):
        '''
        Method to compute the activity coefficient for each component in the model. 

        Parameters:
            x_array (np.ndarray): Liquid mole fraction of each component.
            Temp (float, optional): Temperature at which to calculate activity coefficients.

        Returns:
            np.ndarray: Activity coefficients of each component.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        '''
        #Assert that Lambdas[(i,i)] = 1
        for i in range(1, self.num_comp+1):
            if (self.Lambdas[(i,i)] != 1):
                raise ValueError('Lambda Coefficients entered incorrectly')
            
        gamma_list = []
        for k in range(1, self.num_comp+1):
            gamma_k = 1
            log_arg = 0
            for j in range(1, self.num_comp+1):
               log_arg += ( x_[j-1] * self.Lambdas[(k,j)] )
            if log_arg <= 0:
                raise ValueError
            gamma_k -= np.log(log_arg)

            for i in range(1, self.num_comp+1):
                dividend = (x_[i-1] * self.Lambdas[(i,k)] )
                divisor = 0
                for j in range(1, self.num_comp+1):
                    divisor += (x_[j-1] * self.Lambdas[(i,j)] )
                gamma_k -= (dividend / divisor)
            gamma_list.append(np.exp(gamma_k))
        return np.array(gamma_list)
    
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
