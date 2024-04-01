import numpy as np
import os, sys
#
# Panwa: I'm not sure how else to import these properly
#
PROJECT_ROOT = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            os.pardir)
)
sys.path.append(PROJECT_ROOT) 
from thermo_models.VLEModelBaseClass import *
import matplotlib.pyplot as plt 
from matplotlib import axes
import seaborn as sns
import random as rand
from utils.AntoineEquation import *
from thermo_models.RaoultsLawModel import *
from distillation.DistillationModel import DistillationModel

sns.set_context('talk')

class DistillationModelDoubleFeed(DistillationModel):
    
    def __init__(self, thermo_model:VLEModel, Fr: float, zF: np.ndarray, xFL: np.ndarray, xFU: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, qL = 1, qU = 1) -> None:
        """
        DistillationModelDoubleFeed constructor 
        This class is for extractive distillations using 3+ components
        Args:
            thermo_model (VLEModel): Vapor-Liquid Equilibrium (VLE) model to be used in the distillation process.
            Fr (float): Feed ratio (moles in Feed : moles in Entrainer)
            zF (np.ndarray): Mole fraction of each component in the summation of Feed + Entrainer.
            xFL (np.ndarray): Mole fraction of each component in the feed.
            xFU (np.ndarray): Mole fraction of each component in the entrainer.
            xD (np.ndarray): Mole fraction of each component in the distillate.
            xB (np.ndarray): Mole fraction of each component in the bottom product.
            reflux (Optional): Reflux ratio. If not provided, it will be calculated based on other parameters.
            boil_up (Optional): Boil-up ratio. If not provided, it will be calculated based on other parameters.
            qL (float, optional): Feed condition (q) where q = 1 represents saturated liquid feed and q = 0 represents saturated vapor feed. Defaults to 1.
            qU (float, optional): Entrainer condition (q) where q = 1 represents saturated liquid feed and q = 0 represents saturated vapor feed. Defaults to 1.
            
        """
        D_B  = (zF[0]-xB[0])/(xD[0]-zF[0])
        FL_B = (xD[0]-xB[0])/(Fr*(xD[0]-xFU[0])+xD[0]-xFL[0])
        
        # assume reflux is given, boil_up is not given, and qU and qL are 1.  
        # according to eqn 5.22
        boil_up = ((reflux+1)*D_B)+(FL_B*(Fr*(qU-1))+(qL-1))

        # let self.xF = zF; self.q = qL
        super().__init__(thermo_model,zF,xD,xB,reflux)
        self.xFU = xFU # composition of entrainer fluid entering in upper feed
        self.qU = qU # quality of upper feed
        self.Fr = Fr # feed ratio = (entrainer flow rate)/(feed flow rate)
        self.xFL = xFL
        self.boil_up = boil_up
        self.qL = qL
        self.zF = zF

    """
    The following functions are used to update newly entered parameter values.
    Boilup ratio is dependent on these changes, so it is also updated with each change.    
    """

    def update_boilup(self):
        D_B  = (self.zF[0]-self.xB[0])/(self.xD[0]-self.zF[0])
        FL_B = (self.xD[0]-self.xB[0])/(self.Fr*(self.xD[0]-self.xFU[0])+self.xD[0]-self.xFL[0])
        self.boil_up =  ((self.reflux+1)*D_B)+(FL_B*(self.Fr*(self.qU-1))+(self.qL-1))
        
    def set_Fr(self, Fr_new):
        self.Fr = Fr_new
        self.update_boilup()

    def set_zF(self, zF_new):
        self.zF = zF_new
        self.update_boilup()

    def set_xFL(self, xFL_new):
        self.xFL = xFL_new
        self.update_boilup()

    def set_xFU(self, xFU_new):
        self.xFU = xFU_new
        self.update_boilup()
        
    def set_xD(self, xD_new):
        self.xD = xD_new
        self.update_boilup()
        
    def set_xB(self, xB_new):
        self.xB = xB_new
        self.update_boilup()
        
    def set_r(self, r_new):

        self.reflux = r_new
        self.update_boilup()
        
    def middle_step_y_to_x(self, y_m_j: np.ndarray):
        """
        Method to calculate y in the middle section of the distillation column from given y.

        Args:
            y_m_j (np.ndarray): Mole fraction of each component in the vapor phase in the middle section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        h = (self.Fr*(self.xD[0]-self.xB[0]))/(self.Fr*(self.xFU[0]-self.xB[0])+self.xF[0]-self.xB[0])
        # x = block1 * y_m_k + block2 (eq 5.21a)
        block1 = (self.reflux+1+((self.qU-1)*h))/(self.reflux+(self.qU*h))
        block2 = (h*self.xFU-self.xD)/(self.reflux+self.qU*h)
        
        return (block1*y_m_j)+block2

    def middle_step_x_to_y(self, x_m_j: np.ndarray): 
        """
        Method to calculate y in the middle section of the distillation column from given x.

        Args:
            x_m_j (np.ndarray): Mole fraction of each component in the liquid phase in the middle section.

        Returns:
            np.ndarray: Mole fraction of each component in the vapor phase in the rectifying section that corresponds to x_r_j.
        """
        h = (self.Fr*(self.xD[0]-self.xB[0]))/(self.Fr*(self.xFU[0]-self.xB[0])+self.xF[0]-self.xB[0])

        block1 = (self.reflux+1+((self.qU-1)*h))/(self.reflux+(self.qU*h))
        block2 = (h*self.xFU-self.xD)/(self.reflux+self.qU*h)
        
        return (x_m_j-block2)/block1

       
        
    def plot_rect_comp(self, ax):
        """
        Method to display the rectifying section plot
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
        Returns:
            The graphical output for a demo in interactive examples is produced
        """
        
        x_rect_comp = self.compute_rectifying_stages()[0]
        
        #Extract x1 and x2 from arrays
        x1_rect = x_rect_comp[:, 0]
        x2_rect = x_rect_comp[:, 1]

        # Plot the line connecting the points
        ax.plot(x1_rect, x2_rect, '-D', label='Rectifying Line', color = "#e41a1c", markersize = 6)  # '-o' means a line with circle markers at each data point
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line

        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        ax.legend(loc = 'upper right', fontsize = 12)

        
    def compute_rectifying_stages(self):
        """
        Method to calculate the compositions at each stage in the rectifying section of the column
        Args:
            None
        Returns:
            As an example, this method can be called with the following statement:
            x_rect_comp = self.compute_rectifying_stages()[0]
            The function produced an array of compositions at each stage
        """
        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0
        
        x1 = self.xD
        y1 = self.rectifying_step_xtoy(x1)

        while True:

            x_comp.append(x1) 
            y_comp.append(y1) 

            counter += 1
            x2 = self.thermo_model.convert_y_to_x(y1)[0][:-1]
            y2 = self.rectifying_step_xtoy(x2)
        
            # Loop continues until compositions stop changing or until 200 stages are computed
            if counter == 200:
                print("Too many stages: R OL:", counter)
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 0.0000001:
                return np.array(x_comp), np.array(y_comp)
                
            x1 = x2
            y1 = y2
            
    def compute_stripping_stages(self):
        """
        Method to calculate the compositions at each stage in the stripping section of the column
        Args:
            None
        Returns:
            As an example, this method can be called with the following statement:
            x_strip_comp = self.compute_stripping_stages()[0]
            The function produced an array of compositions at each stage
        """
        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0
        
        x1 = self.xB
        y1 = self.stripping_step_xtoy(x1)

        while True:
            x_comp.append(x1)
            y_comp.append(y1)
            counter += 1
            
            y2 = self.thermo_model.convert_x_to_y(x1)[0][:-1]
            x2 = self.stripping_step_ytox(y2)
            
            # Loop continues until compositions stop changing or until 200 stages are computed
            if counter == 200:
                print("Too many stages: SOL", counter)
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 0.0000000001:
                return np.array(x_comp), np.array(y_comp)
                
            x1 = x2
            y1 = y2
    
    def compute_middle_stages(self, start_point:int):
        """
        Method to calculate the compositions at each stage in the middle section of the column
        Args:
            Start_point (int): Counting from the bottom of the column, the stage number on the stripping section 
            at which the feed is introduced to begin the middle section
        Returns:
            As an example, this method can be called with the following statement:
            x_middle_comp = self.compute_middle_stages(start_point = 5)[0]
            The function produced an array of compositions at each stage
        """
        
        x_comp, y_comp = [], []  # Initialize composition lists
        counter = 0
        
        x_strip_comp = self.compute_stripping_stages()[0]
        x1 = x_strip_comp[start_point, :] # starting point set based on stripping stage number
        y1 = self.middle_step_x_to_y(x1)
        
        while True:
            
            x_comp.append(x1)
            y_comp.append(y1)

            counter += 1          
            y2 = self.thermo_model.convert_x_to_y(x1)[0][:-1]
            x2 = self.middle_step_y_to_x(y2)

            # Loop continues until compositions stop changing or until 200 stages are computed
            # Also stops if compositions leave the physically real regimes of [0,1]            
            if counter == 200:
                print("Too many stages : MS OL:", counter)
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 0.0000000001:
                return np.array(x_comp), np.array(y_comp)
            if (np.any(x2 < 0) or np.any(y2 < 0)):
                return np.array(x_comp), np.array(y_comp)
            
            x1 = x2
            y1 = y2
        

    def plot_rect_strip_comp(self, ax: axes, middle_start):
        """
        Method to display the distillation column with all 3 sections included
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
        Returns:
            The graphical output for a demo in interactive examples is produced
        """

        middle_start = (middle_start - 1) 
        x_rect_comp = self.compute_rectifying_stages()[0]
        x_strip_comp = self.compute_stripping_stages()[0]
        x_middle_comp = self.compute_middle_stages(start_point = middle_start)[0]

        #Extract x1 and x2 from arrays
        x1_rect   = x_rect_comp[:, 0]
        x2_rect   = x_rect_comp[:, 1]
        x1_strip  = x_strip_comp[:, 0]
        x2_strip  = x_strip_comp[:, 1]
        x1_middle = x_middle_comp[:,0]
        x2_middle = x_middle_comp[:,1]
        
        # Plot the line connecting the points
        ax.plot(x1_rect, x2_rect, '-D', label='R OL', color = "#e41a1c", markersize = 6)  
        ax.plot(x1_strip, x2_strip, '-s', label='S OL', color = "#377eb8", markersize = 6)  
        ax.plot(x1_middle, x2_middle, '-s', label='MS OL', color = "#4daf4a", markersize = 6) 

        # Mark special points
        #ax.scatter(self.xF[0], self.xF[1], marker='x', color='orange', label='xF', s = 100)
        #ax.scatter(self.xB[0], self.xB[1], marker='x', color='purple', label='xB', s = 100)
        #ax.scatter(self.xD[0], self.xD[1], marker='x', color='green', label='xD', s = 100)
        #ax.scatter(self.xFL[0], self.xFL[1], marker='x', color='black', label='xFL', s = 100)
        #ax.scatter(self.xFU[0], self.xFU[1], marker='x', color='blue', label='xFU', s = 100)
        
        ax.set_aspect('equal', adjustable='box')

        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(fontsize = 12)


    def plot_strip_comp(self, ax: axes):
        """
        Method to display the stripping section plot
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
        Returns:
            The graphical output for a demo in interactive examples is produced
        """

        x_strip_comp = self.compute_stripping_stages()[0]

        #Extract x1 and x2 from arrays
        x1_strip  = x_strip_comp[:, 0]
        x2_strip  = x_strip_comp[:, 1]
        
        # Plot the line connecting the points
        ax.plot(x1_strip, x2_strip, '-s', label='S OL', color = "#377eb8", markersize = 6)  

        ax.set_aspect('equal', adjustable='box')

        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(fontsize = 12)

    def plot_middle_comp(self, ax: axes, middle_start):
        """
        Method to display the middle section plot
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
            middle_start (int): Stage number where stripping section changes to middle sectiom
        Returns:
            The graphical output for a demo in interactive examples is produced
        """

        middle_start = (middle_start - 1)
        x_middle_comp = self.compute_middle_stages(start_point = middle_start)[0]

        #Extract x1 and x2 from arrays
        x1_middle = x_middle_comp[:,0]
        x2_middle = x_middle_comp[:,1]
        
        # Plot the line connecting the points
        ax.plot(x1_middle, x2_middle, '-s', label='Middle Section', color = "#4daf4a", markersize = 6) 

        # Mark special points
        
        ax.set_aspect('equal', adjustable='box')

        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(fontsize = 12)

        
    def compute_equib_stages(self, ax_num, fixed_points = []):
        pass
        
    def plot_distil(self, ax, ax_fixed):
        pass

    def change_fr(self, new_fr):
        """
        This method updates all parameter which depend on the Feed Ratio
        """

        self.Fr = new_fr
        D_B  = (self.zF[0]-self.xB[0])/(self.xD[0]-self.zF[0])
        FL_B = (self.xD[0]-self.xB[0])/(self.Fr*(self.xD[0]-self.xFU[0])+self.xD[0]-self.xFL[0])
        self.boil_up = ((self.reflux+1)*D_B)+(FL_B*(self.Fr*(self.qU-1))+(self.qL-1))
        return self

       
    def plot_mb(self, ax: axes):
        """
        This method outputs a graphical depiction for a demo of column feasibility
        Args:
            ax (matplotlib.axes.Axes): The axis on which to plot the data.
        Output:
            Graph showing the inlet compositions connected with mass balance lines
        """

        # Mark special points
        ax.scatter(self.zF[0], self.zF[1], marker='X', color='#ff7f00', label='xF', s = 100)
        ax.scatter(self.xB[0], self.xB[1], marker='X', color='#984ea3', label='xB', s = 100)
        ax.scatter(self.xD[0], self.xD[1], marker='X', color='#4daf4a',  label='xD', s = 100)
        ax.scatter(self.xFL[0], self.xFL[1], marker='X', color='#66c2a5', label='xFL', s = 100)
        ax.scatter(self.xFU[0], self.xFU[1], marker='X', color='#fc8d62', label='xFU', s = 100)

        ax.plot([self.xB[0], self.xD[0]], [self.xB[1], self.xD[1]], color = '#2b8cbe', linestyle = 'dasproducinhed')  # Diagonal dashed line
        ax.plot([self.xFL[0], self.xFU[0]], [self.xFL[1], self.xFU[1]], color = '#8da0cb', linestyle = 'dashed')  # Diagonal dashed line
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(loc = 'best', fontsize = 12)
