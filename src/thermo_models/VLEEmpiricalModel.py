import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve
from typing import Callable
import random as rand

PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 

from thermo_models.VLEModelBaseClass  import *

class VLEEmpiricalModelBinary(VLEModel):
    def __init__(self, comp_names, func_xtoy: Callable[[float], float]) -> None:
        """
        Initializes the VLE empirical model for a binary system.

        Args:
            comp_names (list): Names of the components in the binary system.
            func_xtoy (Callable[[float], float]): Function to convert mole fraction x to y.

        Note:
            This model simplifies the VLE calculation by directly using an empirical function for conversion without involving system pressure or partial pressure equations.
        """
        super().__init__(2, None, comp_names, None, False)
        self.func_xtoy = func_xtoy


    def convert_x_to_y(self, x: float):
        """
        Converts mole fraction x in the liquid phase to y in the vapor phase using the provided function.

        Args:
            x (np.ndarray): Mole fraction in the liquid phase.

        Returns:
            solution (np.ndarray): Mole fraction in the vapor phase.
            message (str): Informational message.
        """

        solution = self.func_xtoy(x)
        return solution, "Does not use a solver"
    
    def convert_y_to_x(self, y_array: np.ndarray, x_guess: float = None):
        """Converts y to x by solving the provided function.

        Args:
            y (np.ndarray): Mole fraction in the vapor phase.
            x_guess (float, optional): Initial guess for x. Defaults to a random value between 0 and 1.

        Returns:
            solution (np.ndarray): Mole fraction in the liquid phase.
            message (str): Informational message.

        Raises:
            ValueError: If fsolve does not find a solution or solution is not between 0 and 1.
        """
        # Define a function that needs to be solved
        
        y = y_array[0]
        def func(x):
            return self.func_xtoy(x) - y

        # Initial guess for the liquid mole fractions
        if x_guess is None:
            x_guess = rand.uniform(0,1)
            
        # Iterate until solution is found or max iterations are reached
        for _ in range(500):  # limit the number of iterations
            solution, infodict, ier, mesg = fsolve(func, x_guess, full_output=True, xtol=1e-12)
            if ier == 1:  # fsolve succeeded
                # Check if the solution is valid (i.e., the sum of mole fractions is 1)
                if solution > 1 or solution < 0:
                    raise ValueError("Mole fractions must be between 0 and 1")
                return solution, mesg
            # fsolve failed, generate a new guess
            x_guess = rand.uniform(0,1)

        raise ValueError("fsolve failed to find a solution")
    
    def plot_binary_yx(self, data_points: int = 100) -> None:
        """
        Plot the Vapor-Liquid Equilibrium (VLE) y-x diagram for a binary mixture.

        Args:
            data_points (int, optional): Number of data points to generate for the plot. Defaults to 100.

        """

        x_space = np.linspace(0,1,data_points)
        y_space = [self.convert_x_to_y(x)[0] for x in x_space]

        x_axis = np.linspace(0,1,1000)
        
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(x_space, y_space, label="y vs x")
        ax.plot(x_axis, x_axis, color = 'k', linestyle = '--', label = '45° Line')
        ax.set_xlabel("Mole fraction in liquid phase, x", fontsize = 16)
        ax.set_ylabel("Mole fraction in vapor phase, y" , fontsize = 16)
        plt.title("Vapor-Liquid Equilibrium (VLE) y-x Diagram", fontsize = 16)
        ax.legend(loc = 'best')
        plt.show()
    

    
