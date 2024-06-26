import numpy as np

import matplotlib.pyplot as plt 
import random as rand

from thermo_models import VLEModel
from scipy.optimize import fsolve
from scipy.optimize import brentq


#Notes:
#Conditions for a feasible column, profiles match at the feed stage  + no pinch point in between xB and xD
class DistillationModel:
    
    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        """
        DistillationModel constructor

        Args:
            thermo_model (VLEModel): Vapor-Liquid Equilibrium (VLE) model to be used in the distillation process.
            xF (np.ndarray): Mole fraction of each component in the feed.
            xD (np.ndarray): Mole fraction of each component in the distillate.
            xB (np.ndarray): Mole fraction of each component in the bottom product.
            reflux (float, Optional): Reflux ratio. If not provided, it will be calculated based on other parameters.
            boil_up (float, Optional): Boil-up ratio. If not provided, it will be calculated based on other parameters.
            q (float, optional): Feed condition (q) where q = 1 represents saturated liquid feed and q = 0 represents saturated vapor feed. Defaults to 1.
        
        Raises:
            ValueError: If the reflux, boil-up and q are not correctly specified. Only two of these parameters can be independently set.
        """
        self.thermo_model = thermo_model
        self.num_comp = thermo_model.num_comp
        
        self.xF = xF
        self.xD = xD
        self.xB = xB
        self.q = q
        
        if reflux is not None and boil_up is not None and q is None:
            self.boil_up = boil_up
            self.reflux = reflux
            self.q = ((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0])))-(reflux*((xF[0]-xB[0])/(xD[0]-xB[0]))) #this one need 1 component
        elif reflux is None and boil_up is not None and q is not None:
            self.boil_up = boil_up
            self.q = q
            self.reflux = (((boil_up+1)*((xD[0]-xF[0])/(xD[0]-xF[0]))) - self.q)/((xF[0]-xB[0])/(xD[0]-xB[0])) #this one need 1 component
        elif reflux is not None and boil_up is None and q is not None:
            self.reflux = reflux
            self.q = q
            self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1 #this one need 1 component
        else:
            raise ValueError("Underspecification or overspecification: only 2 variables between reflux, boil up, and q can be provided")
        
    def rectifying_step_xtoy(self, x_r_j:np.ndarray):
        """
        Method to calculate y in the rectifying section of the distillation column from given x.

        Args:
            x_r_j (np.ndarray): Mole fraction of each component in the liquid phase in the rectifying section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        r = self.reflux
        xD = self.xD
        return ((r/(r+1))*x_r_j)+((1/(r+1))*xD)

    def rectifying_step_ytox(self, y_r_j):
        """
        Method to calculate x in the rectifying section of the distillation column from given y.

        Args:
            y_r_j (np.ndarray): Mole fraction of each component in the vapor phase in the rectifying section.

        Returns:
            np.ndarray: Mole fraction of each component in the liquid phase in the rectifying section which corresponds to y_r_j.
        """
        r = self.reflux
        xD = self.xD
        return (((r+1)/r)*y_r_j - (xD/r))
    
    def stripping_step_ytox(self, y_s_j):
        """
        Method to calculate x in the stripping section of the distillation column from given y.

        Args:
            y_s_j (np.ndarray): Mole fraction of each component in the vapor phase in the stripping section.

        Returns:
            np.ndarray: Mole fraction of each component in the liquid phase in the stripping section that corresponds to y_s_j.
        """
        boil_up = self.boil_up
        xB = self.xB
        return ((boil_up/(boil_up+1))*y_s_j)+((1/(boil_up+1))*xB)
    
    def stripping_step_xtoy(self, x_s_j):
        """
        Method to calculate y in the stripping section of the distillation column from given x.

        Args:
            x_s_j (np.ndarray): Mole fraction of each component in the liquid phase in the stripping section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the stripping section.
        """
        boil_up = self.boil_up
        xB = self.xB
        return ((boil_up+1)/boil_up)*x_s_j - (xB/boil_up)
    
    def compute_equib(self):
        """
        Computes the equilibrium stages in the distillation column by iterating over a range of compositions.

        Returns:
            tuple: A tuple containing np.ndarrays of x (liquid phase composition), y (vapor phase composition), and temperature for each stage.
        """
        x1_space = np.linspace(0, 1, 1000)
        y_array = np.zeros((x1_space.size, 2))
        t_array = np.zeros(x1_space.size)
        
        # Initialize numpy arrays
        x_array = np.zeros((x1_space.size, 2))
        for i, x1 in enumerate(x1_space):

            x_array[i] = [x1, 1 - x1]  # Fill the x_array directly
            solution = self.thermo_model.convert_x_to_y(x_array[i])[0]
            y_array[i] = solution[:-1]
            t_array[i] = solution[-1]
            
        return x_array, y_array, t_array
    
    def change_r(self, new_r):
        """
        Updates the reflux ratio and recalculates the boil-up ratio based on the new reflux ratio.

        Parameters:
            new_r (float): New reflux ratio to be set.

        Returns:
            DistillationModel: Self, with updated reflux and boil-up ratios.
        """
        self.reflux = new_r
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
        return self
        
    def set_xD(self, xD_new):
        """
        Updates the distillate composition and recalculates the boil-up ratio based on the new distillate composition.

        Parameters:
            xD_new (np.ndarray): New mole fraction array of distillate composition.

        """

        self.xD = xD_new
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
        
    def set_xB(self, xB_new):
        """
        Updates the bottom product composition and recalculates the boil-up ratio based on the new bottom product composition.

        Parameters:
            xB_new (np.ndarray): New mole fraction array of bottom product composition.

        """
        self.xB = xB_new
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1
        
    def set_r(self, r_new):
        """
        Updates the reflux ratio and recalculates the boil-up ratio based on the new reflux ratio.

        Parameters:
            r_new (float): New reflux ratio to be set.
        """

        self.reflux = r_new
        self.boil_up = ((self.reflux+self.q)*((self.xF[0]-self.xB[0])/(self.xD[0]-self.xF[0]))) + self.q - 1

   
