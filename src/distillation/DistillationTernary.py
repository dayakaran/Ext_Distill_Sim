import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import axes
import random as rand

from thermo_models import VLEModel
from distillation import DistillationModelSingleFeed

sns.set_context('talk')

class DistillationModelTernary(DistillationModelSingleFeed):

    def __init__(self, thermo_model:VLEModel, xF: np.ndarray, xD: np.ndarray, xB: np.ndarray, reflux = None, boil_up = None, q = 1) -> None:
        super().__init__(thermo_model,xF,xD,xB,reflux,boil_up,q)
            
                
    def compute_rectifying_stages(self):

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
            
            #x_comp.append(x2)
            #y_comp.append(y2)
            
            if counter == 100000:
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 1.0e-10:
                return np.array(x_comp), np.array(y_comp)
                
            x1 = x2
            y1 = y2
            

    def compute_stripping_stages(self):

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

            #x_comp.append(x2) # Should this be x2 ?
            #y_comp.append(y2)
                        
            if counter == 5000:
                return np.array(x_comp), np.array(y_comp)
            if np.linalg.norm(x1 - x2) < 1.0e-10:
                return np.array(x_comp), np.array(y_comp)
                
            x1 = x2
            y1 = y2


            
    def plot_rect_strip_comp(self, ax: axes):

        x_rect_comp  = self.compute_rectifying_stages()[0]
        x_strip_comp = self.compute_stripping_stages()[0]
                
        # Plot the line connecting the points
        ax.plot(x_rect_comp[:-1, 0], x_rect_comp[:-1, 1], '-D', label='Rectifying Line', color = "#e41a1c", markersize = 6)  # '-D' means a line with diamond markers at each data point
        ax.plot( x_strip_comp[:-1, 0],  x_strip_comp[:-1, 1], '-s', label='Stripping Line', color = "#377eb8", markersize = 6)  # '-s' means a line with box markers at each data point
        ax.plot( x_rect_comp[-1, 0],  x_rect_comp[-1, 1], '*', label='OL Terminus', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint
        ax.plot( x_strip_comp[-1, 0],  x_strip_comp[-1, 1], '*', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint

        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='X', color='#ff7f00', label='xF', s = 100)
        ax.scatter(self.xB[0], self.xB[1], marker='X', color='#984ea3', label='xB', s = 100)
        ax.scatter(self.xD[0], self.xD[1], marker='X', color='#4daf4a',  label='xD', s = 100)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(loc = 'upper right', fontsize = 12)


    def plot_strip_comp(self, ax: axes):

        x_strip_comp = self.compute_stripping_stages()[0]
                
        ax.plot( x_strip_comp[:-1, 0],  x_strip_comp[:-1, 1], '-s', label='Stripping Line', color = "#377eb8", markersize = 6)  # '-s' means a line with box markers at each data point
        ax.plot( x_strip_comp[-1, 0],  x_strip_comp[-1, 1], '*', label='OL Terminus', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint

        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='X', color='#ff7f00', label='xF', s = 100)
        ax.scatter(self.xB[0], self.xB[1], marker='X', color='#984ea3', label='xB', s = 100)
                
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(loc='upper right', fontsize = 12)


    def plot_rect_comp(self, ax: axes):

        x_rect_comp  = self.compute_rectifying_stages()[0]
        
        # Plot the line connecting the points
        ax.plot(x_rect_comp[:-1, 0], x_rect_comp[:-1, 1], '-D', label='Rectifying Line', color = "#e41a1c", markersize = 6)  # '-D' means a line with diamond markers at each data point
        ax.plot( x_rect_comp[-1, 0],  x_rect_comp[-1, 1], '*', label='OL Terminus', color = "black", markersize = 15)  # '-*' means a line with a star marker at the endpoint
        
        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='X', color='#ff7f00', label='xF', s = 100)
        ax.scatter(self.xD[0], self.xD[1], marker='X', color='#4daf4a',  label='xD', s = 100)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(loc = 'upper right', fontsize = 12)

   
    def plot_mb(self, ax: axes):

        # Mark special points
        ax.scatter(self.xF[0], self.xF[1], marker='X', color='#ff7f00', label='xF', s = 100)
        ax.scatter(self.xB[0], self.xB[1], marker='X', color='#984ea3', label='xB', s = 100)
        ax.scatter(self.xD[0], self.xD[1], marker='X', color='#4daf4a',  label='xD', s = 100)

        ax.plot([self.xB[0], self.xD[0]], [self.xB[1], self.xD[1]], color = '#2b8cbe', linestyle = 'dashed')  # Diagonal dashed line
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        
        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        
        ax.legend(loc = 'best', fontsize = 12)
