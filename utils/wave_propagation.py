import numpy as np
from tqdm import tqdm

def pml_damping_profile(
        num_coordinates: int,
        num_pml_points: int,
        max_damping: float,
):
        """
        Computes the damping profile for a Perfectly Matched Layer (PML).

        This function generates a damping profile for the PML boundary in a finite 
        difference wave simulation. The damping profile is applied to the boundary 
        to absorb outgoing waves and prevent reflections. The profile starts from 
        zero at the interior of the model and increases towards the boundary, where 
        it reaches the maximum damping value.

        Args:
                num_coordinates (int): The number of spatial points in the model (i.e., the 
                        number of grid points in the spatial domain).
                num_pml_points (int): The number of points at the boundary where the PML 
                        damping is applied.
                max_damping (float): The maximum damping value at the outer boundary of the 
                        PML.

        Returns:
                np.ndarray: A 1D array of size `num_coordinates` representing the damping 
                        profile, with values increasing from zero to `max_damping` at the boundary.

        Example:
                damping = pml_damping_profile(
                num_coordinates=100,
                num_pml_points=10,
                max_damping=0.1
                )
        
        Notes:
                - The damping is applied in a linear fashion starting from zero at the 
                interior of the model and gradually increasing to `max_damping` at the 
                boundary defined by `num_pml_points`.
                - The function assumes that the PML region is located at the end of the model.
        """
        
        damping = np.zeros((num_coordinates,))

        # PML damping profile
        for i in range(num_pml_points):
                damping[- (i + 1)] = max_damping * (num_pml_points - i) / num_pml_points

        return damping


def finite_difference_solver(
        velocity_models: np.ndarray,
        num_coordinates: int,
        num_time_steps: int,
        delta_x: float,
        delta_t: float,
        source: np.ndarray,
        source_location: int,
        damping: np.ndarray,
):
        """
        Solves a finite difference model for wave propagation with an absorbing boundary.

        This function implements a finite difference method to solve the wave equation 
        for displacement in a 1D medium, and includes a perfectly matched layer (PML) boundary 
        for wave absorption at the bottom boundary. The solver computes the displacement field over a specified 
        number of time steps and tracks the peak amplitudes of the displacement for each 
        velocity model.

        Args:
                velocity_models (np.ndarray): A 2D array of velocity models with shape 
                        (num_velocity_models, num_coordinates), where each row represents the 
                        velocity values at each coordinate point for a given model.
                num_coordinates (int): The number of spatial points in the model (i.e., the 
                        number of grid points in the spatial domain).
                num_time_steps (int): The number of time steps for the simulation.
                delta_x (float): The spatial step size.
                delta_t (float): The time step size.
                source (np.ndarray): A 1D array representing the source time function, where 
                        each entry corresponds to the source value at a given time step.
                source_location (int): The index of the spatial location of the source in the model.
                damping (np.ndarray): A 1D array representing the damping profile for the PML 
                        absorbing boundary. It should have the same size as `num_coordinates`.

        Returns:
                Tuple[np.ndarray, np.ndarray]:
                - peak_amplitudes (np.ndarray): A 2D array of shape (num_velocity_models, 
                        num_time_steps), where each entry corresponds to the peak amplitude of 
                        the displacement field for a given velocity model and time step.
                - u_plot (np.ndarray): A 2D array of shape (num_time_steps, num_coordinates) 
                        representing the displacement field over time for a single velocity model 
                        (for plotting).
        
        Notes:
                - The function uses a second-order Runge-Kutta method (Ralston's method) 
                for time integration on the absorbing boundary auxilliary variables, 
                and three-point centred difference finite differences for time integration 
                on the wavefield.
                - The PML (Perfectly Matched Layer) boundary is applied to absorb outgoing 
                waves at the boundaries of the model.
                - A free-surface boundary condition is applied at the surface 
                (x = 0), where the displacement gradient is zero.
        """        
        
        rk_alpha = 2. / 3.  # 2nd-order Runge-Kutta alpha parameter (2/3 corresponds to Ralston's method)

        num_velocity_models = velocity_models.shape[0]  # number of velocity models

        u = np.zeros((num_velocity_models, num_coordinates))  # initialise displacement field
        u_old = np.zeros((num_velocity_models, num_coordinates)) # initialise previous displacement field

        d2udx2 = np.zeros((num_velocity_models, num_coordinates))    # initialise second spatial derivative of displacement field

        # initialise auxiliary variables for the PML absorbing boundary
        phi = np.zeros((num_velocity_models, num_coordinates))
        psi = np.zeros((num_velocity_models, num_coordinates))

        # more auxiliary variables for the PML absorbing boundary
        phi_1 = np.zeros((num_velocity_models, num_coordinates))
        psi_1 = np.zeros((num_velocity_models, num_coordinates))
        phi_2 = np.zeros((num_velocity_models, num_coordinates))
        psi_2 = np.zeros((num_velocity_models, num_coordinates))

        # Runge-Kutta 2nd-order time integration for the PML absorbing boundary
        # initialise RK slopes for auxiliary variables
        k1_phi = np.zeros((num_velocity_models, num_coordinates))
        k2_phi = np.zeros((num_velocity_models, num_coordinates))
        k1_psi = np.zeros((num_velocity_models, num_coordinates))
        k2_psi = np.zeros((num_velocity_models, num_coordinates))

        phi_psi = np.zeros((num_velocity_models, num_coordinates))

        damping_stack = np.stack([damping] * num_velocity_models, axis=0) # stack the damping profile for each velocity model

        peak_amplitudes = np.zeros((num_velocity_models, num_time_steps))   # initialise peak amplitudes of the displacement field for each velocity model
        u_plot = np.zeros((num_time_steps, num_coordinates))  # initialise array to store plotting data for a single velocity model

        plot_idx = 1    # index of the velocity model to plot the displacement field for

        for it in tqdm(range(num_time_steps)):

                for i in range(1, num_coordinates - 1):

                        # calculate the second spatial derivative of the displacement field
                        # three-point centred difference
                        d2udx2[:, i] = (u[:, i + 1] - 2 * u[:, i] + u[:, i - 1]) / delta_x ** 2

                        # calculate the auxiliary variables for the PML absorbing boundary
                        phi_1[:, i] = - (damping_stack[:, i - 1] * phi[:, i - 1] + damping_stack[:, i] * phi[:, i]) / 2.
                        phi_2[:, i] = - (u[:, i + 1] - u[:, i - 1]) / (2. * delta_x)
                        psi_1[:, i] = - (damping_stack[:, i - 1] * psi[:, i] + damping_stack[:, i] * psi[:, i + 1]) / 2.
                        psi_2[:, i] = - (u[:, i + 1] - u[:, i - 1]) / (2. * delta_x)

                k1_phi = phi_1 + phi_2
                k1_psi = psi_1 + psi_2

                for i in range(1, num_coordinates - 1):

                        k2_phi[:, i] = - (damping_stack[:, i - 1] * (phi[:, i - 1] + rk_alpha * delta_t * k1_phi[:, i - 1]) + damping_stack[:, i] * (phi[:, i] + rk_alpha * delta_t * k1_phi[:, i])) / 2. - (u[:, i + 1] - u[:, i - 1]) / (2. * delta_x)
                        k2_psi[:, i] = - (damping_stack[:, i - 1] * (psi[:, i] + rk_alpha * delta_t * k1_psi[:, i]) + damping_stack[:, i] * (psi[:, i + 1] + rk_alpha * delta_t * k1_psi[:, i + 1])) / 2. - (u[:, i + 1] - u[:, i - 1]) / (2. * delta_x)

                # RK time integration for auxiliary variables
                phi_new = phi + delta_t * ((1 - 1 / (2 * rk_alpha)) * k1_phi + 1 / (2 * rk_alpha) * k2_phi)
                psi_new = psi + delta_t * ((1 - 1 / (2 * rk_alpha)) * k1_psi + 1 / (2 * rk_alpha) * k2_psi)

                # add damping for absorbing boundary
                for i in range(1, num_coordinates - 1):

                        phi_psi[:, i] = damping_stack[:, i] * psi_new[:, i + 1] - damping_stack[:, i - 1] * phi_new[:, i - 1]

                # time integration for displacement field, including absorbing boundary term
                # three-point centred difference
                u_new = 2 * u - u_old + velocity_models ** 2 * delta_t ** 2 * (d2udx2 + phi_psi / delta_x)

                # free surface boundary condition at x = 0
                # acoustic pressure is zero at the surface, therefore the divergence of the displacement field is zero
                u_new[:, 0] = u_new[:, 1]

                # updating source term
                u_new[:, source_location] = u_new[:, source_location] + source[it] / delta_x * delta_t ** 2

                # update variables
                u_old, u = u, u_new
                phi_old, phi = phi, phi_new
                psi_old, psi = psi, psi_new

                # update the peak amplitudes of the displacement field for each velocity model
                peak_amplitudes[:, it] = np.maximum(peak_amplitudes[:, it - 1], np.abs(u[:, 0]))

                # update the plotting data for a single velocity model
                u_plot[it, :] = u[plot_idx, :]

        return peak_amplitudes, u_plot
