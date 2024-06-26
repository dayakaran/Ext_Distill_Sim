import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import random as rand

from utils.rand_comp_gen import *

class VLEModel:
    """
    Base class for vapor-liquid equilibrium models.
    
    """
    def __init__(self, num_comp: int, P_sys: float, comp_names, partial_pressure_eqs, use_jacobian=False):
        """
        Initializes the VLEModel object with specified parameters.

        Args:
            num_comp (int): Number of components in the system
            P_sys (float): System pressure
            comp_names (list): Names of components in the system
            partial_pressure_eqs (list): A list containing AntoineEquationBase10 or AntoineEquationBaseE objects for each component, used to calculate their partial pressures.
            use_jacobian (bool, optional): Flag if jacobian should be used in optimizations. Defaults to False.
        """
        self.num_comp = num_comp
        self.P_sys = P_sys
        self.comp_names = comp_names
        self.use_jacobian = use_jacobian
        self.partial_pressure_eqs = partial_pressure_eqs
        
    def get_activity_coefficient(self, x_array, Temp = None)->np.ndarray:
        """
        Method to compute the activity coefficient for each component in the model. 
        Must be implemented by subclasses.

        Parameters:
            x_array (np.ndarray): Liquid mole fraction of each component.
            Temp (float, optional): Temperature at which to calculate activity coefficients.

        Returns:
            np.ndarray: Activity coefficients of each component.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """

        raise NotImplementedError
    
    def get_vapor_pressure(self, Temp)->np.ndarray:
        """
        Method to compute the vapor pressure for each component in the model. Must be implemented by subclasses.

        Parameters:
            Temp (float): Temperature at which to calculate vapor pressures.

        Returns:
            np.ndarray: vapor pressure of each component
            
        Raises:
            NotImplementedError: Not implemented for base class
            
        """
        raise NotImplementedError
    
    def convert_x_to_y(self, x_array:np.ndarray, temp_guess = None)->np.ndarray:
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (np.ndarray): Liquid mole fraction of each component.
            temp_guess (float): inital temperature guess for fsolve

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The solution from the fsolve function, including the vapor mole fractions and the system temperature.
                - str: A message describing the exit condition of fsolve.
        """
        
        # Compute the boiling points for each component
        boiling_points = [eq.get_boiling_point(self.P_sys) for eq in self.partial_pressure_eqs]

        #Provides a random guess for temp if no temp_guess was provided as a parameter
        if temp_guess == None:
            temp_guess = rand.uniform(np.amax(boiling_points), np.amin(boiling_points))

        # Use fsolve to find the vapor mole fractions and system temperature that satisfy the equilibrium conditions
        ier = 0 #fsolve results, 1 for convergence, else nonconvergence
        runs = 0
        while True:
            runs += 1
            if runs % 1000 == 0:
                print("Current Run from convert_x_to_y:",runs)
            try:
                random_number = generate_point_system_random_sum_to_one(self.num_comp) #generate random composition as intial guess
                new_guess = np.append(random_number,temp_guess) #create initial guess for composition and temperature
               
                #use fsolve with jacobian if provided
                if self.use_jacobian: 
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy, new_guess, args=(x_array,), full_output=True, xtol=1e-12, fprime=self.jacobian_x_to_y)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
                else:
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy, new_guess, args=(x_array,), full_output=True, xtol=1e-12, fprime=None)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
            except:
                continue

            
    def convert_y_to_x(self, y_array:np.ndarray, temp_guess = None)->np.ndarray:
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (np.ndarray): Vapor mole fraction of each component.
            temp_guess (float, optional): Initial temperature guess for fsolve. If not provided, a random temperature within a logical range is used.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The solution from the fsolve function, including the liquid mole fractions and the system temperature.
                - str: A message describing the exit condition of fsolve.

        Note:
            This method attempts to solve the system until convergence, using fsolve. It utilizes an iterative approach, potentially leveraging the Jacobian if 'use_jacobian' is set to True and a corresponding Jacobian function is provided.
        """

        
        # Compute the boiling points for each component
        boiling_points = [eq.get_boiling_point(self.P_sys) for eq in self.partial_pressure_eqs]
        
        #Provides a random guess for temp if no temp_guess was provided as a parameter
        if temp_guess == None:
            temp_guess = rand.uniform(np.amax(boiling_points), np.amin(boiling_points))
        
        #Parallel to convert_x_to_y, refer to comments above        
        ier = 0
        runs = 0
        while True:

            runs += 1 

            if runs % 10000 == 0:
                print("Current Run from convert_y_to_x:",runs)
            try:            
                random_number = generate_point_system_random_sum_to_one(self.num_comp)
                new_guess     = np.append(random_number, temp_guess)
                
                if self.use_jacobian:
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy2, new_guess, args=(y_array,), full_output=True, xtol=1e-12, fprime=self.jacobian_y_to_x)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
                else:
                    solution, infodict, ier, mesg = fsolve(self.compute_Txy2, new_guess, args=(y_array,), full_output=True, xtol=1e-12, fprime=None)
                    if not np.all(np.isclose(infodict["fvec"],0,atol = 1e-8)):
                        raise ValueError("Not converged")
                    if ier == 1:
                        return solution, mesg
            except:
                continue
        
    def compute_Txy(self, vars:np.ndarray, x_array:np.ndarray)->list:
        """
        Computes the system of equations for the T-x-y calculations for convert_x_to_y.

        This function is used as the input to the fsolve function to find the roots 
        of the system of equations, which represent the equilibrium conditions for 
        the vapor-liquid equilibrium calculations.

        Args:
            vars (np.ndarray): A 1D array containing the initial guess for the vapor mole fractions and the system temperature.
            x_array (np.ndarray): A 1D array containing the liquid mole fractions.

        Returns:
            eqs (list): A list of the residuals of the equilibrium equations
        """
        
        # Extract the vapor mole fractions and temperature from the vars array
        y_array = vars[:-1]
        Temp = vars[-1]
        # Compute the left-hand side of the equilibrium equations
        lefths = x_array * self.get_activity_coefficient(x_array, Temp=Temp) * self.get_vapor_pressure(Temp)

        # Compute the right-hand side of the equilibrium equations
        righths = y_array * self.P_sys

        # Form the system of equations by subtracting the right-hand side from the left-hand side
        # Also include the normalization conditions for the mole fractions
        eqs = (lefths - righths).tolist() + [np.sum(y_array) - 1]

        return eqs
    
    def compute_Txy2(self, vars:np.ndarray, y_array:np.ndarray)->list:
        """
        Computes the system of equations for the T-x-y calculations for convert_y_to_x.

        This function is used as the input to the fsolve function to find the roots 
        of the system of equations, which represent the equilibrium conditions for 
        the vapor-liquid equilibrium calculations.

        Args:
            vars (np.ndarray): A 1D array containing the initial guess for the liquid mole fractions and the system temperature.
            y_array (np.ndarray): A 1D array containing the vapor mole fractions.

        Returns:
            eqs (list):  A list of the residuals of the equilibrium equations.
        """
        
        # Extract the liquid mole fractions and temperature from the vars array
        x_array = vars[:-1]
        Temp = vars[-1]

        # Compute the left-hand side of the equilibrium equations
        lhs = x_array * self.get_activity_coefficient(x_array, Temp=Temp) * self.get_vapor_pressure(Temp)
        
        # Compute the right-hand side of the equilibrium equations
        rhs = y_array * self.P_sys

        # Form the system of equations by subtracting the right-hand side from the left-hand side
        # Also include the normalization conditions for the mole fractions
        eqs = (lhs - rhs).tolist() + [np.sum(x_array) - 1]

        return eqs


    def plot_binary_Txy(self, data_points:int, comp_index:int, ax):
        """
        Plots the T-x-y diagram for a binary mixture on the given ax object.

        Args:
            data_points (int): Number of data points to use in the plot.
            comp_index (int): Index of the component to plot.
            ax (matplotlib.axes._axes.Axes): The matplotlib axis object to plot on.

        Raises:
            ValueError: If the number of components is not 2.
        """
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary mixtures.")

        # Create an array of mole fractions for the first component
        x1_space = np.linspace(0, 1, data_points)

        # Create a 2D array of mole fractions for both components
        x_array = np.column_stack([x1_space, 1 - x1_space])

        # Initialize lists to store the vapor mole fractions and system temperatures
        y_array, t_evaluated = [], []

        # Compute the vapor mole fractions and system temperatures for each set of liquid mole fractions
        for x in x_array:
            solution = self.convert_x_to_y(x)[0]
            y_array.append(solution[:-1])
            t_evaluated.append(solution[-1])

        # Convert the list of vapor mole fractions to a 2D numpy array
        y_array = np.array(y_array)

        # Use the passed ax object for plotting
        ax.plot(x_array[:, comp_index], t_evaluated, label="Liquid phase")
        ax.plot(y_array[:, comp_index], t_evaluated, label="Vapor phase")
        ax.set_title("T-x-y Diagram for " + self.__class__.__name__)
        ax.set_xlabel(f"Mole fraction of component {comp_index + 1}")
        ax.set_ylabel("Temperature")
        ax.legend()

    def plot_yx_binary(self, data_points:int=100):
        """
        Plots the y-x diagram for a binary mixture.
        
        Args:
            data_points (int, optional): Number of data points to use in the plot. Default is 100.
        """
        
        # Initialize a figure for plotting
        plt.figure(figsize=(10, 6))
        
        # Create an array of mole fractions for the first component
        x_space = np.linspace(0, 1, data_points)
        y_space = []
        
        # Compute the vapor mole fractions for each set of liquid mole fractions
        for x1 in x_space:
            # The mole fraction of the second component in the liquid phase is 1 - x1
            x2 = 1 - x1
            # Initialize a mole fraction array for all components
            x_array = np.array([x1, x2])
            # Solve for the vapor-liquid equilibrium
            y_array = self.convert_x_to_y(x_array)[0]
            # Append the vapor mole fraction for the first component
            y_space.append(y_array[0])

        # Plot y vs. x for the binary mixture
        plt.plot(x_space, y_space, label="Component 1")
        plt.xlabel("Liquid phase mole fraction (Component 1)")
        plt.ylabel("Vapor phase mole fraction (Component 1)")
        plt.title("y-x Diagram for Binary Mixture")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        
   

                
            
        
        


        
