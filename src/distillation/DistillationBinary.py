import numpy as np
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from thermo_models.VLEModelBaseClass import *
import matplotlib.pyplot as plt 
import random as rand
from scipy.optimize import fsolve
from scipy.optimize import brentq
from utils.AntoineEquation import *
from thermo_models.RaoultsLawModel import *
from distillation.DistillationSingleFeed import DistillationModelSingleFeed

import seaborn as sns
sns.set_context("poster")
sns.set_style("ticks")

class DistillationModelBinary(DistillationModelSingleFeed):
    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        """
        DistillationModelBinary constructor 
        This class is for simple binary distillations
        Args:
            thermo_model (VLEModel): Vapor-Liquid Equilibrium (VLE) model to be used in the distillation process.
            xF (np.ndarray): Mole fraction of each component in the feed.
            xD (np.ndarray): Mole fraction of each component in the distillate.
            xB (np.ndarray): Mole fraction of each component in the bottom product.
            reflux (Optional): Reflux ratio. If not provided, it will be calculated based on other parameters.
            boil_up (Optional): Boil-up ratio. If not provided, it will be calculated based on other parameters.
            q (float, optional): Feed condition (q) where q = 1 represents saturated liquid feed and q = 0 represents saturated vapor feed. Defaults to 1.
        
        Raises:
            ValueError: If the reflux, boil-up and q are not correctly specified. Only two of these parameters can be independently set.
            With a defaul value for q=1, either boil_up or reflux should be specified and the other can be calculated.
        """

        super().__init__(thermo_model,xF,xD,xB,reflux,boil_up,q)     
        self.x_array_equib, self.y_array_equib, self.t_array = self.compute_equib() 
        
        # Initialize numpy arrays
        self.y_s_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))
        self.y_r_array = np.zeros((self.x_array_equib[:, 0].size, self.thermo_model.num_comp))
            
    def find_rect_fixedpoints_binary(self, n):
        """
        Method to calculate fixed points in the rectifying section.

        Args:
            n (int): number of steps toward convergence (typically use n=30)

        Returns:
            x0_values, y0_values (1d.array): Arrays for the liquid and vapor mole fractions of component 0 at the fixed points
        """

        rand.seed(0)
        x0_values = []
        y0_values = []

        def compute_rect_fixed(x_r):
            sol_array, mesg = self.thermo_model.convert_x_to_y(np.array([x_r, 1 - x_r]))
            y_x_0 = sol_array[0]
            return - y_x_0 + self.rectifying_step_xtoy(x_r_j=x_r)[0]

        # Define the initial bracket
        a = 0

        # Generate a list of n points in the range [0, 1]
        partition_points = np.linspace(0.0001, 0.9999, n+1)[1:]  # We start from the second point because the first is 0

        for b in partition_points:
            try:
                x_r_0 = brentq(compute_rect_fixed, a, b, xtol=1e-5)
                y_r_0 = self.thermo_model.convert_x_to_y(np.array([x_r_0, 1- x_r_0]))[0][0]
                x0_values.append(x_r_0)
                y0_values.append(y_r_0)
            except ValueError:
                pass
            # Update a to be the current b for the next partition
            a = b

        return x0_values, y0_values 
    
    
    def find_strip_fixedpoints_binary(self, n):
        """
        Method to calculate fixed points in the stripping section.

        Args:
            n (int): number of steps toward convergence (typically use n=30)

        Returns:
            x0_values, y0_values (1d.array): Arrays for the liquid and vapor mole fractions of
            the most volatile component 0 at the fixed points
        """


        rand.seed(0)
        x0_values = []
        y0_values = []
        
        def compute_strip_fixed(x_s):            
            # Check if x_s is zero or very close to zero
            if abs(x_s) < 1e-10:
                return float('inf')
            else:
                sol_array, mesg = self.thermo_model.convert_x_to_y(np.array([x_s, 1 - x_s]))
                y_s_0 = sol_array[0]
                return y_s_0 - self.stripping_step_xtoy(x_s_j = x_s)[0]

        # Define the initial bracket
        a = 0

        # Generate a list of n points in the range [0, 1]
        partition_points = np.linspace(0.0001, 0.9999, n+1)[1:]  # We start from the second point because the first is 0

        for b in partition_points:
            try:
                x_s_0 = brentq(compute_strip_fixed, a, b, xtol=1e-8)
                y_s_0 = self.thermo_model.convert_x_to_y(np.array([x_s_0, 1 - x_s_0]))[0][0]
                y0_values.append(y_s_0)
                x0_values.append(x_s_0)
            except ValueError:
                pass
            # Update a to be the current b for the next partition
            a = b

        return x0_values, y0_values


    def compute_equib_stages_binary(self, ax_num, fixed_points = []):
        """
        Method to calculate the compositions at each stage in a binary distillation columm

        Args:
            ax_num (int): 0, 1, or 2 depending on the graph being analyzed
            fixed_points (1d.array): List of liquid mole fractions at fixed points for component 0

        Returns:
            x_comp (1d.array): Array for the liquid mole fractions of component 0 at each stage
            y_comp (1d.array): Array for the vapor mole fractions of component 0 at each stage
            N (int): number of stages in the column

        Raises:
            ValueError: if mole fractions move outide of the physically real range [0,1]
            ValueError: if the number of components is not 2
            ValueError: if ax_num is not equal to 0, 1, or 2.
        """

        ## ADD POINTS TO X AXIS TO REPRESENT NUMBER OF EQUILIBRIA ##
        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")

        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0

        if ax_num == 0:
            N = 0  # Track number of equib stages
            x1 = self.xB[0]
            y1 = self.stripping_step_xtoy(np.array([x1, 1 - x1]))[0]

            while x1 < self.xD[0]:
                if x1 > 1 or y1 > 1 or x1 < 0 or y1 < 0:
                    raise ValueError("Components out of range")

                # Convergence stops occuring at tangent pinches
                if np.isclose(x1, fixed_points, rtol=0.001).any():
                    return x_comp, y_comp, "Infinite Stages"

                # This is equivalent to the McCabe-Thiele graphical method
                # "Step" from leaving stages in equilibrium to the next point on operating line    
                counter += 1   
                if counter > 200:
                    return x_comp, y_comp, "Too many stages > 200"

                x_comp.append(x1)
                y_comp.append(y1)
                N += 1

                y2 = self.thermo_model.convert_x_to_y(np.array([x1, 1-x1]))[0][0]
                x_comp.append(x1)
                y_comp.append(y2)

                x2 = self.stripping_step_ytox(np.array([y2, 1 - y2]))[0]  # Step to x value on operating line
                x1, y1 = x2, y2

        elif ax_num in [1, 2]:
            N = 0  # Track number of equib stages
            x1 = self.xD[0]
            y1 = self.rectifying_step_xtoy(np.array([x1, 1 -x1]))[0]

            while x1 > self.xB[0]:
                if x1 > 1 or y1 > 1 or x1 < 0 or y1 < 0:
                    raise ValueError("Components out of range")
                counter += 1
                if np.isclose(x1, fixed_points, atol=0.001).any():
                    return x_comp, y_comp, "Infinite Stages"
                if counter > 200:
                    return x_comp, y_comp, "Too many stages > 200"

                x_comp.append(x1)
                y_comp.append(y1)
                N += 1

                x2 = self.thermo_model.convert_y_to_x(np.array([y1, 1-y1]))[0][0]
                x_comp.append(x2)
                y_comp.append(y1)

                yr = self.rectifying_step_xtoy(np.array([x2, 1 - x2]))[0]
                ys = self.stripping_step_xtoy(np.array([x2, 1 - x2]))[0]
                y2 = min(yr, ys)
                x1 = x2
                counter += 1

                y1 = yr if ax_num == 1 else y2
        else:
            raise ValueError("This method only accepts ax_num = 0,1,2.")


        return x_comp, y_comp, N
    
    def plot_distil_strip_binary(self, ax):
        """
        Method to display the stripping section plot
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
        Returns:
            The graphical output for a demo in interactive examples is produced
        """

        # Set limits for all main plots
        ax.set_xlim([0,1])
        ax.set_ylim([-0.05,1.05])

        # Plot the x - axis
        ax.hlines(0, 0, 1, linewidth = 1, color = 'k', linestyle = 'dotted')
        
        # Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])
        
        for i, x1 in enumerate(self.x_array_equib):
            self.y_s_array[i] = self.stripping_step_xtoy(x1)
        
        # Plot the stripping line
        s_min_index = int(1000 * self.xB[0])
        for i in range(len(self.y_s_array)):
            if (self.y_s_array[i,0] > self.y_array_equib[i,0]):
                s_max_index = i
                break 
    
        ax.plot(self.x_array_equib[s_min_index:s_max_index, 0], self.y_s_array[s_min_index:s_max_index, 0], color = 'green')

        
        # Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')
        ax.vlines(self.xB[0], 0, self.xB[0], linestyles = 'dashed', colors = 'k')
        
        # Plot the fixed points from stripping line
        self.x_s_fixed, self.y_s_fixed = self.find_strip_fixedpoints_binary(n=30)

        ax.scatter(self.x_s_fixed, self.y_s_fixed, s=50, c="red")
        x_fixed, y_fixed, N_1 = self.compute_equib_stages_binary(0, self.x_s_fixed)

        ax.plot(x_fixed, y_fixed, linestyle='--', color='red', alpha = 0.3)        
        ax.scatter(self.x_s_fixed, self.y_s_fixed, s=50, c="red")

        # Plot fixed point along the x - axis and the stagewise composition      
        ax.scatter(self.x_s_fixed, [0]*len(self.x_s_fixed), marker='o', color='red', s = 100)
        ax.scatter(x_fixed, [0]*len(x_fixed), marker='^', color='blue', facecolors='none', edgecolors='blue', linewidths = 0.75, s = 40)
        
        ax.tick_params(axis='x', which='both', labelsize = 18, width = 2, length = 4)
        ax.tick_params(axis='y', which='both', labelsize = 18, width = 2, length = 4)

        ax.set_title("Stripping Section", fontsize = 20)
        species = self.thermo_model.comp_names[0]

        x_label = '$x_{' + species + '}$'
        y_label = '$y_{' + species + '}$'

        ax.set_xlabel(x_label, labelpad = 35, fontsize = 22)
        ax.set_ylabel(y_label, labelpad = 10, fontsize = 22)

        return 
    
    def plot_distil_rect_binary(self, ax, rect_title = "Rectifying Section"):
        """
        Method to display the rectifying section plot
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
            rect_title (string): title name
        Returns:
            The graphical output for a demo in interactive examples is produced
        Raises:
            ValueError: if the number of components is not 2
        """

        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
        
        ax.set_xlim([0,1])
        ax.set_ylim([-0.05,1.05])

        # Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])

        # Plot the x - axis
        ax.hlines(0, 0, 1, linewidth = 1, color = 'k', linestyle = 'dotted')
        
        ax.vlines(self.xD[0], 0, self.xD[0], linestyles = 'dashed', colors = 'k')

        for i, x1 in enumerate(self.x_array_equib):
            self.y_r_array[i] = self.rectifying_step_xtoy(x1)
        
        # Plot the rectifying line
        r_max_index = int(1000 * self.xD[0])
        for i in range(len(self.y_r_array)):
            if (self.y_r_array[i,0] < self.y_array_equib[i,0]):
                r_min_index = i
                break 
        ax.plot(self.x_array_equib[r_min_index:r_max_index, 0], self.y_r_array[r_min_index:r_max_index, 0], color = 'green')
        
        # Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')

        # Plot the fixed points from stripping line
        self.x_r_fixed, self.y_r_fixed = self.find_rect_fixedpoints_binary(n=30)
        
        x_stages, y_stages, N_2 = self.compute_equib_stages_binary(1, self.x_r_fixed)
        ax.plot(x_stages, y_stages, linestyle='--', color='blue', alpha = 0.3)
        ax.scatter(self.x_r_fixed, self.y_r_fixed, s=50, c="red")

        # Plot fixed point along the x - axis and the stagewise composition     
        ax.scatter(self.x_r_fixed, [0]*len(self.x_r_fixed), marker='o', color='red', s = 100)
        ax.scatter(x_stages, [0]*len(x_stages), marker='^', color='blue', facecolors='none', edgecolors='blue', linewidths = 0.75, s = 40)
        ax.tick_params(axis='x', which='both', labelsize = 18, width = 2, length = 4)
        ax.tick_params(axis='y', which='both', labelsize = 18, width = 2, length = 4)
        ax.set_title(rect_title, fontsize = 20)

        species = self.thermo_model.comp_names[0]
        x_label = '$x_{' + species + '}$'
        y_label = '$y_{' + species + '}$'

        ax.set_xlabel(x_label, labelpad = 35, fontsize = 22)
        ax.set_ylabel(y_label, labelpad = 10, fontsize = 22)

        return
    

        
    def plot_distil_binary(self, ax):
        """
        Method to display the full distillation column for binary mixtures
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
        Returns:
            The graphical output for a demo in interactive examples is produced
        Raises:
            ValueError: if the number of components is not 2
        """

        if self.num_comp != 2:
            raise ValueError("This method can only be used for binary distillation.")
        
                # Set limits for ax
        ax.set_xlim([0,1])
        ax.set_ylim([-0.05,1.05])
        
        for i, x1 in enumerate(self.x_array_equib):
            self.y_r_array[i] = self.rectifying_step_xtoy(x1)
            self.y_s_array[i] = self.stripping_step_xtoy(x1)

        # Plot the x - axis
        ax.hlines(0, 0, 1, linewidth = 1, color = 'k', linestyle = 'dotted')
            
        #Plot the equilibrium curve
        ax.plot(self.x_array_equib[:, 0], self.y_array_equib[:, 0])

        # Plot top and bottom compositions
        ax.vlines(self.xB[0], 0, self.xB[0], linestyles = 'dashed', colors = 'k')
        ax.vlines(self.xD[0], 0, self.xD[0], linestyles = 'dashed', colors = 'k')

        op_color             = 'green'
        intersection_counter = 0
        
        for i in range(len(self.y_r_array)-1, 0, -1): #Iterate backwards starting at top of rectifying curve
            if (abs((self.y_r_array[i,0]) - self.y_array_equib[i,0]) <= 0.001): #rectifying line intersects equib
                intersection_counter += 1
            if (abs((self.y_r_array[i,0]) - self.y_s_array[i,0]) <= 0.001): # operating lines interect
                if ((self.y_r_array[i,0] < self.y_array_equib[i,0]) & (intersection_counter > 1)):
                    op_color = 'black'
                if (self.y_r_array[i,0] >= self.y_array_equib[i,0]): #intersection occurs above equilibrium curve
                    op_color = 'red'  
                break 
               
        #Plot the rectifying line
        r_max_index = int(1000 * self.xD[0])
        for i in range(len(self.y_r_array)):
            if (self.y_r_array[i,0] < self.y_array_equib[i,0]): #set min index where the rectifying line intersects operating line
                r_min_index = i
                break 
        
        
        #Plot the stripping line
        s_min_index = int(1000 * self.xB[0])
        for i in range(len(self.y_s_array)): #set max index where stripping line intersects equib curve
            if (self.y_s_array[i,0] > self.y_array_equib[i,0]):
                s_max_index = i
                break 
            
        # For valid columns with intersecting operating lines, make the segments after the intersection occurs be lighter in color
        if (op_color == 'green'): # meaning the column params are valid
            for i in range(len(self.y_r_array)):
                if (self.y_r_array[i,0] < self.y_s_array[i,0]): #find index where operating lines intersect
                    op_intersect_index = i
                    break 
            # Partitiion the operating lines based on the intersection, increase line transparency
            ax.plot(self.x_array_equib[op_intersect_index:r_max_index, 0], self.y_r_array[op_intersect_index:r_max_index, 0], color = op_color)
            ax.plot(self.x_array_equib[s_min_index:op_intersect_index, 0], self.y_s_array[s_min_index:op_intersect_index, 0], color = op_color)  
            ax.plot(self.x_array_equib[op_intersect_index:s_max_index, 0], self.y_s_array[op_intersect_index:s_max_index, 0], color = op_color, alpha = 0.3) 
            ax.plot(self.x_array_equib[r_min_index:op_intersect_index, 0], self.y_r_array[r_min_index:op_intersect_index, 0], color = op_color, alpha = 0.3)         
        else:
            ax.plot(self.x_array_equib[r_min_index:r_max_index, 0], self.y_r_array[r_min_index:r_max_index, 0], color = op_color)
            ax.plot(self.x_array_equib[s_min_index:s_max_index, 0], self.y_s_array[s_min_index:s_max_index, 0], color = op_color)
        
        
        #Plot y = x line
        ax.plot([0,1], [0,1], linestyle='dashed')

        x_r_0, y_r_0 = self.find_rect_fixedpoints_binary(n=30)
        x_s_0, y_s_0 = self.find_strip_fixedpoints_binary(n=30)
        
        x_stages, y_stages, N_2 = self.compute_equib_stages_binary(2, x_r_0 + x_s_0)
        if (op_color == 'green'):
            ax.plot(x_stages, y_stages, linestyle='--', color='green', alpha = 0.3)
        else:
            xr_stages, yr_stages, N_r = self.compute_equib_stages_binary(1, self.x_r_fixed)
            ax.plot(xr_stages, yr_stages, linestyle='--', color='blue', alpha = 0.3)
            xs_stages, ys_stages, N_s = self.compute_equib_stages_binary(0, self.x_s_fixed)
            ax.plot(xs_stages, ys_stages, linestyle='--', color='red', alpha = 0.3)

        ax.scatter(x_r_0 + x_s_0, y_r_0 + y_s_0, s=100, c="red")

        x_rect = self.compute_equib_stages_binary(1, x_r_0 + x_s_0)[0]
        ax.scatter(x_rect, [0]*len(x_rect), marker='o', color='blue', facecolors='none', edgecolors='blue', linewidths = 0.75, s = 40)
        x_strip = self.compute_equib_stages_binary(0, x_r_0 + x_s_0)[0]
        ax.scatter(x_strip, [0]*len(x_strip), marker = '^', color = 'blue', facecolors='none', edgecolors='blue', linewidths = 0.75, s = 40)
        ax.text(0.5, 0.9, f"No Stages: {N_2}", ha='center', va='center', transform=ax.transAxes, fontsize = 16)

        #Graphical Formatting
        ax.scatter(self.x_r_fixed, [0]*len(self.x_r_fixed), marker='o', color='red', s = 50)
        ax.scatter(self.x_s_fixed, [0]*len(self.x_s_fixed), marker='o', color='red', s = 50)

        ax.tick_params(axis='x', which='both', labelsize = 18, width = 2, length = 4)
        ax.tick_params(axis='y', which='both', labelsize = 18, width = 2, length = 4)
        ax.set_title("Full Column", fontsize = 20)
        species = self.thermo_model.comp_names[0]

        x_label = '$x_{' + species + '}$'
        y_label = '$y_{' + species + '}$'

        ax.set_xlabel(x_label, labelpad = 35, fontsize = 20)
        ax.set_ylabel(y_label, labelpad = 10, fontsize = 22)

        return 
