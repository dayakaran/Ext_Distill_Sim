import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import random as rand

from utils.rand_comp_gen import *

#: Default number of random restarts allowed before a flash is declared infeasible.
#: Each restart reseeds fsolve from a fresh random composition, so the attempt count
#: needed is a heavy-tailed random variable rather than a fixed cost. Measured on the
#: ternary Margules system, convert_y_to_x has a median of 1 attempt but a p90 of ~86
#: and a worst case of ~1200; convert_x_to_y converges almost immediately. 5000 leaves
#: roughly 4x headroom over the worst case observed while still guaranteeing the call
#: terminates -- an infeasible input costs ~7s here and ~30s under Pyodide, rather than
#: hanging forever with no way to interrupt it in a browser.
DEFAULT_MAX_ATTEMPTS = 5000


class VLEConvergenceError(RuntimeError):
    """
    Raised when a flash calculation fails to converge within max_attempts restarts.

    Because each restart is seeded randomly, an occasional failure is not proof of
    infeasibility -- but exhausting a budget this large is overwhelmingly likely to
    mean the requested composition is not attainable for this model.
    """


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
    
    def _solve_flash(self, residual_func, known_array:np.ndarray, jacobian, temp_guess, max_attempts:int, label:str):
        """
        Solves an equilibrium flash by random-restarting fsolve until it converges.

        Shared implementation behind convert_x_to_y and convert_y_to_x, which differ
        only in which residual function, Jacobian and known composition they supply.

        Args:
            residual_func (callable): Residual system to drive to zero (compute_Txy or compute_Txy2).
            known_array (np.ndarray): The known phase composition passed through to residual_func.
            jacobian (callable or None): Analytic Jacobian, or None to let fsolve approximate it.
            temp_guess (float or None): Initial temperature guess; randomised between the
                component boiling points when None.
            max_attempts (int): Maximum number of random restarts before giving up.
            label (str): Caller name, used in the error message.

        Returns:
            tuple: (solution, mesg) as returned by fsolve.

        Raises:
            VLEConvergenceError: If no restart converges within max_attempts.
        """
        # Compute the boiling points for each component
        boiling_points = [eq.get_boiling_point(self.P_sys) for eq in self.partial_pressure_eqs]

        # Provides a random guess for temp if no temp_guess was provided as a parameter
        if temp_guess is None:
            temp_guess = rand.uniform(np.amax(boiling_points), np.amin(boiling_points))

        last_error = None

        for _ in range(max_attempts):
            try:
                # Generate a random composition as the initial guess, plus the temperature
                random_number = generate_point_system_random_sum_to_one(self.num_comp)
                new_guess = np.append(random_number, temp_guess)

                solution, infodict, ier, mesg = fsolve(
                    residual_func, new_guess, args=(known_array,), full_output=True,
                    xtol=1e-12, fprime=jacobian,
                )

                if ier == 1 and np.all(np.isclose(infodict["fvec"], 0, atol=1e-8)):
                    return solution, mesg

                last_error = mesg
            except Exception as exc:
                # A restart from an unlucky guess can legitimately blow up; keep trying.
                last_error = exc

        raise VLEConvergenceError(
            f"{label} failed to converge for {np.array2string(known_array, precision=4)} "
            f"after {max_attempts} restarts. This composition is most likely infeasible "
            f"for {self.__class__.__name__} at P_sys={self.P_sys}. "
            f"Last solver report: {last_error}"
        )

    def convert_x_to_y(self, x_array:np.ndarray, temp_guess = None, max_attempts:int = DEFAULT_MAX_ATTEMPTS)->np.ndarray:
        """
        Computes the conversion from liquid mole fraction to vapor mole fraction.

        Args:
            x_array (np.ndarray): Liquid mole fraction of each component.
            temp_guess (float): inital temperature guess for fsolve
            max_attempts (int, optional): Maximum number of random restarts before the
                composition is declared infeasible. Defaults to DEFAULT_MAX_ATTEMPTS.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The solution from the fsolve function, including the vapor mole fractions and the system temperature.
                - str: A message describing the exit condition of fsolve.

        Raises:
            VLEConvergenceError: If no restart converges within max_attempts.
        """
        return self._solve_flash(
            self.compute_Txy, x_array,
            self.jacobian_x_to_y if self.use_jacobian else None,
            temp_guess, max_attempts, "convert_x_to_y",
        )

    def convert_y_to_x(self, y_array:np.ndarray, temp_guess = None, max_attempts:int = DEFAULT_MAX_ATTEMPTS)->np.ndarray:
        """
        Computes the conversion from vapor mole fraction to liquid mole fraction.

        Args:
            y_array (np.ndarray): Vapor mole fraction of each component.
            temp_guess (float, optional): Initial temperature guess for fsolve. If not provided, a random temperature within a logical range is used.
            max_attempts (int, optional): Maximum number of random restarts before the
                composition is declared infeasible. Defaults to DEFAULT_MAX_ATTEMPTS.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The solution from the fsolve function, including the liquid mole fractions and the system temperature.
                - str: A message describing the exit condition of fsolve.

        Raises:
            VLEConvergenceError: If no restart converges within max_attempts.
        """
        return self._solve_flash(
            self.compute_Txy2, y_array,
            self.jacobian_y_to_x if self.use_jacobian else None,
            temp_guess, max_attempts, "convert_y_to_x",
        )
        
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
        
        
   

                
            
        
        


        
