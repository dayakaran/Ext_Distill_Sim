import numpy as np

from thermo_models.VLEModelBaseClass import *
from utils.AntoineEquation import *
from distillation.DistillationModel import *
from distillation.DistillationDoubleFeed import *
import seaborn as sns

class PhasePortraits():
    """
    Class for visualizing phase portraits and vector fields for distillation residue curves.
    Contains methods for plotting:
    - Phase vector fields showing direction and magnitude of composition change.
    - Residue curves by integrating the composition dynamics.
    - Phase portraits for stripping, rectifying, and middle (double feed) sections.
    The dynamics are determined by the provided thermo and distillation models.
    """

    def __init__(self, thermo_model:VLEModel, distil_model:DistillationModel = None):
        """
        Initialize PhasePortraits class.
        Parameters
        ----------
        thermo_model (VLEModel): Thermodynamic model to use for VLE calculations. 
        distil_model (DistillationModel): Distillation model to use for dynamics. Defaults to None for no distillation dynamics.
        """

        self.distil_model = distil_model
        self.thermo_model = thermo_model

    def plot_phase_vector_fields(self, ax, dxdt, grid_data_points=20, title = 'Phase Vector Field with Magnitude Colored Arrows'):
        """
        Plots a phase vector field for the given dynamics function dxdt.
        The vector field shows the direction and magnitude of composition
        change at each point in the composition space. Useful for visualizing
        phase portraits and residue curve dynamics.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot the vector field on.
        dxdt : callable
            Dynamics function that takes a composition point
            and returns the rate of change.
        grid_data_points : int, optional
            Number of grid points to sample.
        title : str, optional
            Title for the plot. Defaults to 'Phase Vector Field with Magnitude Colored Arrows'.
        Returns
        -------
        None
        """

        x_array = [np.array(point) for point in create_restricted_simplex_grid(3, grid_data_points)]
        vectors = np.zeros((len(x_array), 2))
        valid_points = []

        for i, x in enumerate(x_array):
            try:
                vector = dxdt(x)
                vectors[i] = vector[:2]
                if not (np.isinf(vectors[i]).any() or np.isnan(vectors[i]).any()):
                    valid_points.append(True)
                else:
                    valid_points.append(False)
            except Exception as e:
                print(f"An error occurred at point {x}: {e}")
                vectors[i] = np.nan
                valid_points.append(False)

        valid_x_array = [x for i, x in enumerate(x_array) if valid_points[i]]
        valid_vectors = np.array([vectors[i] for i in range(len(vectors)) if valid_points[i]])
        magnitudes    = np.linalg.norm(valid_vectors, axis=1)
        norm          = plt.Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())
        cmap          = sns.color_palette("icefire", as_cmap=True)

        for point, vector in zip(valid_x_array, valid_vectors):
            # Correct color mapping for each vector
            vector_magnitude = np.linalg.norm(vector)
            color            = cmap(norm(vector_magnitude))
            ax.quiver(point[0], point[1], vector[0], vector[1], color=color)

        # Set the limits and title for your plot
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line

        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad = 10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)
        ax.set_title(title)

    def plot_vector_field_strip(self, ax, grid_data_points=20):
        """
        Plot the stripping vector field.
        Plots a vector field showing the direction and magnitude of composition
        change during the stripping section. This illustrates the
        stripping process dynamics.
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Matplotlib axis to plot on.
        grid_data_points : int, optional
            Number of grid points to use in each dimension. More points means
            a smoother plot. Defaults to 20.
        Returns
        -------
        None
        """

        def dxdt(x):
            """
            Defines the vector field for stripping section.
            Args:
               x (np.array): composition point array
            Returns:
                np.array: vector field at point x
            """

            try:

                return  self.distil_model.stripping_step_ytox(self.thermo_model.convert_x_to_y (x)[0][:-1]) - x

            except OverflowError:

                print("Overflow occurred in dxdt.")

                return None

        self.plot_phase_vector_fields(ax, dxdt, grid_data_points, title = "Stripping Phase Plane")

    def plot_vector_field_rect(self, ax, grid_data_points=20):

        """

        Plot the rectifying vector field.

        Plots a vector field showing the direction and magnitude of composition

        change during for the rectifying section. This illustrates the

        rectifying process dynamics.

        Parameters

        ----------

        ax: matplotlib.axes.Axes

            Matplotlib axis to plot on.

        grid_data_points: int, optional

            Number of grid points to use in each dimension. More points means a smoother plot. Default to 20

        Returns

        ----------

        None

        """

        def dxdt(x):

            """

            Defines the vector field for rectifying section.

            Args:

                x: composition point

            Returns:

                np.array: vector field at point x

            """

            try:

                return self.thermo_model.convert_y_to_x ( self.distil_model.rectifying_step_xtoy(x) )[0][:-1] - x

            except OverflowError:

                print("Overflow occurred in dxdt.")

                return None

        self.plot_phase_vector_fields(ax, dxdt,grid_data_points, title = "Rectifying Phase Plane")

    def plot_vector_field_residue(self, ax, grid_data_points=20):

        """

        Plots the residue curve vector field.

        Plots a vector field showing the direction and magnitude of composition

        change during the residue curve calculation. This illustrates the

        batch distillation process dynamics.

        Parameters

        ----------

        ax: matplotlib.axes.Axes

            Matplotlib axis to plot on.

        grid_data_points: int, optional

            Number of grid points to use in each dimension. More points means a smoother plot. Default to 20

        """       

        def dxdt(x):

            """

            defines the vector field for a batch distillation.

            Args:

                x: composition point

            Returns:

                np.array: vector field at point x

            """

            try:

                return x - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]

            except OverflowError:

                print("Overflow occurred in dxdt.")

                return None

        self.plot_phase_vector_fields(ax,dxdt,grid_data_points, title = "Residue Phase Plane")


        
    def plot_vector_field_middle(self, ax, grid_data_points=20):

        """

        Plots the middle section vector field.

        Plots a vector field showing the direction and magnitude of composition

        change for middle section. This illustrates the

        middle section process dynamics.

        Parameters

        ----------

        ax: matplotlib.axes.Axes

            Matplotlib axis to plot on.

        grid_data_points: int, optional

            Number of grid points to use in each dimension. More points means a smoother plot. Default to 20

        """    

        # checks if distillation model has a middle section

        if self.distil_model is None:

            raise TypeError("Invalid operation")

        if not isinstance(self.distil_model, DistillationModelDoubleFeed):

            raise TypeError("Invalid operation")

        def dxdt(x):

            """

            defines the vector field for the middle section.

            Args:

                x: composition point

            Returns:

                np.array: vector field at point x

            """

            try:

                return  self.distil_model.middle_step_y_to_x(self.thermo_model.convert_x_to_y (x)[0][:-1]) - x

            except OverflowError:

                print("Overflow occurred in dxdt.")

                return None

        self.plot_phase_vector_fields(ax,dxdt,grid_data_points, title = "Middle Phase Plane")

    def plot_residue_curve(self, ax, t_span, data_points: int = 15, init_comps = None):

        """

        Plots the residue curve on a specified Matplotlib axis by integrating over a given time span to track composition changes.

        This method illustrates the evolution of compositions in a batch distillation process from specified starting compositions.

        Integration proceeds both forwards and backwards from each starting composition to map out the complete residue curve trajectory.

        Parameters

        ----------

        ax : matplotlib.axes.Axes

            The Matplotlib axis on which the residue curve is to be plotted.

        t_span : list or tuple

            The time span over which to integrate, denoted as [start, end]. This parameter defines the range over which the integration occurs.

        data_points : int, optional

            The number of data points to use for plotting the integration path. It dictates the resolution of the plot. Defaults to 15.

        init_comps : list of np.array, optional

            A list of initial composition arrays from which the integration starts.

        Returns

        -------

        None

        """

        def dxdt(t, x):

            """

            Defines the vector field for a batch distillation.

            Args:

                x (np.array): composition point array

            Returns:

                np.array: vector field at point x

            """

            try:

                return x - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]

            except OverflowError:

                print("Overflow occurred in dxdt.")

                return None

        for init_comp in init_comps:

            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)

            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)

        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line

        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)

    def plot_strip_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):

        """

        Plots the stripping curve on a specified Matplotlib axis by integrating over a given time span to track composition changes.

        This method illustrates the evolution of compositions in a stripping process from specified starting compositions.

        Integration proceeds both forwards and backwards from each starting composition to map out the complete stripping curve trajectory.

        Parameters

        ----------

        ax : matplotlib.axes.Axes

            The Matplotlib axis on which the stripping curve is to be plotted.

        t_span : list or tuple

            The time span over which to integrate, denoted as [start, end]. This parameter defines the range over which the integration occurs.

        data_points : int, optional

            The number of data points to use for plotting the integration path. It dictates the resolution of the plot. Defaults to 15.

        init_comps : list of np.array, optional

            A list of initial composition arrays from which the integration starts. Each array within the list represents a specific composition point in the phase space.

        Returns

        -------

        None

        """

        if self.distil_model is None:

            raise TypeError("Invalid operation")

        def dxdt(t, x):

            try:

                return self.distil_model.stripping_step_xtoy(x_s_j=x) - self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]

            except OverflowError:

                print("Overflow occurred in dxdt.")

                return None

        for init_comp in init_comps:

            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)

            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)

        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line

        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)

    def plot_rect_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):

        """

        Plots the rectifying curve on a specified Matplotlib axis by integrating over a given time span to track composition changes.

        This method illustrates the evolution of compositions in a rectifying process from specified starting compositions.

        Integration proceeds both forwards and backwards from each starting composition to map out the complete rectifying curve trajectory.

        Parameters

        ----------

        ax : matplotlib.axes.Axes

            The Matplotlib axis on which the rectifying curve is to be plotted.

        t_span : list or tuple

            The time span over which to integrate, denoted as [start, end]. This parameter defines the range over which the integration occurs.

        data_points : int, optional

            The number of data points to use for plotting the integration path. It dictates the resolution of the plot. Defaults to 15.

        init_comps : list of np.array, optional

            A list of initial composition arrays from which the integration starts. Each array within the list represents a specific composition point in the phase space.

        Returns

        -------

        None

        """

        if self.distil_model is None:

            raise TypeError("Invalid operation")

        def dxdt(t, x):

            try:

                return x - self.distil_model.rectifying_step_ytox(self.thermo_model.convert_x_to_y(x)[0][:-1])

            except OverflowError:

                print("Overflow occurred in dxdt.")

                return None

        for init_comp in init_comps:

            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)

            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)

        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line

        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10)

    def plot_middle_portrait(self, ax, t_span, data_points: int = 15, init_comps = None):

        """

        Plots the middle section curve on a specified Matplotlib axis by integrating over a given time span to track composition changes.

        This method illustrates the evolution of compositions in the middle section of a distillation process from specified starting compositions.

        Integration proceeds both forwards and backwards from each starting composition to map out the complete trajectory for the middle section.

        Parameters

        ----------

        ax : matplotlib.axes.Axes

            The Matplotlib axis on which the middle section curve is to be plotted.

        t_span : list or tuple

            The time span over which to integrate, denoted as [start, end]. This parameter defines the range over which the integration occurs.

        data_points : int, optional

            The number of data points to use for plotting the integration path. It dictates the resolution of the plot. Defaults to 15.

        init_comps : list of np.array, optional

            A list of initial composition arrays from which the integration starts. Each array within the list represents a specific composition point in the phase space.

        Returns

        -------

        None

        """

        if self.distil_model is None:

            raise TypeError("Invalid operation")

        if not isinstance(self.distil_model, DistillationModelDoubleFeed):

            raise TypeError("Invalid operation")

        def dxdt(t, x):

            try:

                return -self.distil_model.middle_step_x_to_y(x) + self.thermo_model.convert_x_to_y(x_array=x)[0][:-1]

            except OverflowError:

                print("Overflow occurred in dxdt.")
                return None

        for init_comp in init_comps:

            self.int_plot_path(ax,init_comp, t_span, data_points,dxdt=dxdt)
            self.int_plot_path(ax, init_comp, [t_span[0],-t_span[1]], data_points, dxdt=dxdt)

        ax.set_aspect('equal', adjustable='box')
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])

        ax.plot([1, 0], [0, 1], 'k--')  # Diagonal dashed line
        ax.hlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line
        ax.vlines(0, 0, 1, colors = 'k', linestyles = 'dashed')  # dashed line

        ax.set_xlabel(self.thermo_model.comp_names[0], labelpad=10)
        ax.set_ylabel(self.thermo_model.comp_names[1], labelpad = 10) 

    def int_plot_path(self, ax, initial, t_span, num_points, dxdt):
        """
        Plots an integrated path for a batch distillation process on a given Matplotlib axis.
        This function visualizes the integration path by plotting both a continuous line and directional arrows along the path.
        Arrows are added every 7 points to indicate the direction of progression, with their orientation reflecting the integration's direction
        (forward or reverse in time/heat).
        Args:
            ax (matplotlib.axes.Axes):
                The Matplotlib axis object where the integrated path will be plotted.
            initial (np.array):
                The initial composition or state from which to start the integration.
            t_span (tuple/list):
                A 2-element sequence specifying the start and end points for the integration.
            num_points (int):
                The number of points to generate along the integrated path.
            dxdt (function):
                A function representing the differential equations to be integrated.
        """

        # Generate the path data using int_path
        path_data = self.int_path(initial, t_span, num_points, dxdt)

        # Determine the direction of time progression
        time_direction = np.sign(t_span[1] - t_span[0])

        # Plot arrows every 7 points along the path
        for i in range(0, len(path_data)-1, 7):
            dx = path_data[i+1][0] - path_data[i][0]
            dy = path_data[i+1][1] - path_data[i][1]

            # Reverse the arrow direction if time is going backwards
            if time_direction < 0:
                dx, dy = -dx, -dy
            ax.arrow(path_data[i][0], path_data[i][1], dx, dy, head_width=0.02, head_length=0.02, fc='k', ec='k')

        # Plot the path as a red line
        ax.plot(path_data[:, 0], path_data[:, 1], color='red')

    def int_path(initial, t_span, num_points, dxdt):

        """

        Integrates the differential equations using the Runge-Kutta 4th order method to generate a path from an initial state over a specified span.

        This function calculates the state of the system at a series of points along the specified time span, using the RK4 method for numerical integration.

        The integration halts prematurely if the state reaches non-physical values (e.g., NaN, infinity, or outside [0, 1] bounds).

        Args:

            initial (np.array):

                The initial state of the system. This should be a NumPy array corresponding to the initial values of the variables being integrated.

            t_span (tuple or list):

                A 2-element sequence specifying the start and end points (inclusive) for the integration.

            num_points (int):

                The number of points at which the state is calculated, including the initial state. Determines the step size of the integration.

            dxdt (function):

                The derivative function to be integrated. This function should take the current state and time as arguments and return the derivative of the state with respect to time.

        Returns:

            np.array:

                An array of states representing the path of the system from the initial state, calculated at the points specified by `t_span` and `num_points`. The shape of the array is (N, M), where N is the number of successful integration steps (may be less than `num_points` if integration is halted) and M is the number of variables in the system.

        """

        x0 = np.array(initial)

        dt = (t_span[1] - t_span[0]) / num_points

        t_eval = np.linspace(t_span[0], t_span[1], num_points)

        x_vals = [x0]

        x = x0

        for t in t_eval:

            x = rk4_step(t, x, dt, dxdt)

            if x is None or np.isinf(x).any() or np.isnan(x).any() or (x > 1).any() or (x < 0).any():

                print("Integration stopped due to overflow, NaN values, or out-of-bound values.")

                break

            x_vals.append(x)

        return np.array(x_vals)

def rk4_step(t, x, dt, dxdt):

    """

    Performs a single step of the Runge-Kutta 4th order method to solve ordinary differential equations.

    Parameters

    ----------

    t: float

        The current time at which the step begins.

    x: np.array

        The current state of the system, represented as a NumPy array of the variables being integrated.

    dt: float

        The time step size for the integration.

    dxdt: function

        The derivative function that calculates the rate of change of the state.

    Returns

    -------

    np.array

        The state of the system after a time step `dt`, represented as a NumPy array. Returns `None` if an `OverflowError` occurs during computation.

    Raises

    ------

    OverflowError

        Indicates a potential numerical instability or an issue with the input values, occurring during the computation of the RK4 steps.

    """

    try:

        k1 = dt * dxdt(t, x)
        k2 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k1)
        k3 = dt * dxdt(t + 0.5 * dt, x + 0.5 * k2)
        k4 = dt * dxdt(t + dt, x + k3)

        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    except OverflowError:

        print("Overflow occurred during integration.")
        return None
