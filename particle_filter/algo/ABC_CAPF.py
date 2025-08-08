import numpy as np
from scipy.stats import levy_stable
# from .trajectory.trajectory_list import TrajectoryList, copy_trajectory_group
from .cython_code.ABC_CAPF import TrajectoryList, copy_trajectory_group, append_trajectory_group

# from particle_filter.algo.trajectory.trajectory_list import TrajectoryList, copy_trajectory_group
# ABC_CAPF.py

class ABC_CAPF:
    def __init__(self, observations, tau, phi, sigma_h2, input_trajectory, epsilon, Z_params, particle_count):
        self.observations = observations
        self.tau = tau
        self.phi = phi
        self.sigma_h2 = sigma_h2
        self.input_trajectory = input_trajectory
        self.epsilon = epsilon
        self.particle_count = particle_count       
        self.Z_params = Z_params
        self.trajectory_size = len(input_trajectory)

    def normalize_particle_weights(self, particles):
        total_weight = np.sum(particles['weight'])
        particles['weight'] /= total_weight
        return particles

    def compute_tempered_weight(self, particles, obs_t):
        # The sqrt_expression is independent of each particle
        sqrt_expression = np.sqrt((np.pi**2) / (self.sigma_h2 + np.pi**2))

        # Perform the likelihood approximations calculation in a vectorized manner
        likelihood_approximations = (
            1 + (
                ((obs_t**2) * sqrt_expression) *
                np.exp(-1 * sqrt_expression * (self.tau + self.phi * np.log(particles['state'])))
            )) ** -1

        # Update weights in the particles structured array
        particles['weight'] *= likelihood_approximations
        
    def resample_particles(self):
        # Normalize the weights of the particles (assuming the method returns a structured numpy array)
        normalized_particles = self.normalize_particle_weights(self.particles[:-1])
        # Extract final normalized weights for resampling
        weights = normalized_particles['weight']
        # Resample indices based on the final normalized weights
        resampled_indices = np.random.choice(len(weights), size=len(weights), p=weights, replace=True)

        trajectories = copy_trajectory_group(self.particles['trajectory'][resampled_indices])
        # Use direct indexing to allocate resampled particles, avoiding one-by-one copying
        self.particles[:-1] = self.particles[:-1][resampled_indices]
        # Handle the trajectory resampling; assume trajectory data needs individual handling
        self.particles['trajectory'][:-1] = trajectories



    def transition_particles(self, t_index):
        # Generate random epsilon values for all particles at once, excluding the last particle
        epsilon_ts = np.random.normal(loc=0, scale=1, size=(self.particle_count - 1))

        # Vectorized state update for all but the last particle
        updated_states = np.exp(self.tau + self.phi * np.log(self.particles['state'][:-1]) + np.sqrt(self.sigma_h2) * epsilon_ts)

        # Update states for all but the last particle
        self.particles['state'][:-1] = updated_states

        # Convert updated states to a NumPy array for efficient batch append
        updated_states_view = np.asarray(updated_states, dtype=np.float64)

        # Update trajectories for all but the last particle using batch append
        append_trajectory_group(np.asarray(self.particles['trajectory'][:-1], dtype=object), updated_states_view)

        # Update the last particle's state and trajectory separately
        last_state = self.input_trajectory[t_index + 1]
        self.particles['state'][-1] = last_state
        self.particles['trajectory'][-1].append(last_state)

    
    def generate_aux_data(self):
        alpha, beta, gamma, delta = self.Z_params

        # Directly generate and assign auxiliary data
        Zt_values = levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=self.particle_count)
        np.multiply(np.sqrt(self.particles['state']), Zt_values, out=self.particles['aux_data'])
    
    def reweight_particles_with_aux_data(self, particles, obs_t):

        # Compute K_ep directly using the 'aux_data' field from particles
        discrepancy = np.abs(obs_t - particles['aux_data'])

        # Compute K_epsilon vectorized. Using Cauchy Kernel to mitigate numerical instability.
        # Using self.epsilon as an error parameter, as if it were a gaussian kernel.
        
        # K_epsilon = 1 / (np.pi * self.epsilon * (1 + (discrepancy / self.epsilon) ** 2))

        K_epsilon = 1 / (1 + (discrepancy**2 / self.epsilon**2 ))

        # Vectorized weight update for all but the last particle
        # Note: This assumes ancestor weights have been correctly set beforehand
        new_weights = (particles['ancestor_weight'][:-1] / particles['weight'][:-1]) * K_epsilon[:-1]

        # Apply the new weights to all but the last particle
        particles['weight'][:-1] = new_weights
        
        # New particle is = K_ep
        particles['weight'][-1] *= K_epsilon[-1]
        
        return particles
    
    def draw_index_b(self):
        # Extract final weights for all particles
        weights = np.array([particle['weight'] for particle in self.particles], dtype=np.float64)
        # Draw an index with probability proportional to the normalized weights
        index_b = np.random.choice(a=range(self.particle_count), size=1, p=weights)
        return index_b[0]
    
#     draw most likely, always
    def draw_index_b(self):
        # Directly access 'weight' field and find the index of the maximum weight
        most_likely_index = np.argmax(self.particles['weight'])
        return most_likely_index
    
#   For debugging:
    def print_particles(self):
        for particle in self.particles:
            state = particle['state']
            weight = particle['weight']
            trajectory = particle['trajectory']  # Retrieve the TrajectoryList object
            aux_data = particle['aux_data']
            ancestor_weight = particle['ancestor_weight']

            # Convert the TrajectoryList to a NumPy array
            trajectory_data = trajectory.to_list()

            # Format the string to include both trajectory values and memory location
            trajectory_info = f"Data: {trajectory_data}, Memory Loc: {id(trajectory)}"

            # Print all information including the formatted trajectory_info
            print(f"State: {state}, Weight: {weight}, Trajectory: {trajectory_info}, Aux Data: {aux_data}, Ancestor Weight: {ancestor_weight}")
    
    def run(self):
            # Simulate initial state for each particle (including the last one for now)
            initial_states = np.random.lognormal(
                mean=self.tau / (1 - self.phi), 
                sigma=self.sigma_h2 / (1 - self.phi**2), 
                size=self.particle_count
            )

            # Adjust the state of the last particle before initializing the structured array
            initial_states[-1] = self.input_trajectory[0]

            dtype = [
                ('state', np.float64),       # For storing the state of each particle
                ('weight', np.float64),      # For storing the weight of each particle
                ('trajectory', 'O'),          # For storing the trajectory object
                ('aux_data', np.float64),  # For storing aux data
                ('ancestor_weight', np.float64), # For storing the ancestor weight of each particle
            ]

            # Initialize the structured array with the defined dtype
            self.particles = np.zeros(self.particle_count, dtype=dtype)

            # Populate the structured array
            self.particles['state'] = initial_states  # Set states
            self.particles['weight'] = np.full(self.particle_count, 1 / self.particle_count)  # Set weights

            # Initialize each trajectory with the initial state of the particle 
            for i in range(self.particle_count):
                #             self.particles['trajectory'][i] = TrajectoryList([initial_states[i]])
                self.particles['trajectory'][i] = TrajectoryList(max_size = self.trajectory_size, initial_data=initial_states[i])

            # print('init'), self.print_particles()

            # Iterate through t
            for t_index, obs_t in enumerate(self.observations):

                self.particles['ancestor_weight'][:-1] = self.particles['weight'][:-1]
#                 print('anc update'), self.print_particles()

                # Compute weights wrt the new observation at time t
                self.compute_tempered_weight(self.particles, obs_t)
#                 print('weight tempering'), self.print_particles()

                # Resample weight indices so that larger weights are probably persisted

                self.resample_particles()
#                 print('resample'), self.print_particles()

                # Transition states of selected particles (those with bigger weights)
                self.transition_particles(t_index)
#                 print('state transition'), self.print_particles()
#                 print('input traj',self.input_trajectory)

                # Draw auxiliary data
                self.generate_aux_data()
#                 print('aux data'), self.print_particles()

                # Update particle weights wrt auxiliary data
                self.particles = self.reweight_particles_with_aux_data(self.particles, obs_t)
#                 print('aux reweighting'), self.print_particles()

                self.particles = self.normalize_particle_weights(self.particles)

            index_b = self.draw_index_b()
            chosen_trajectory = self.particles[index_b]['trajectory']
#             print('final'), self.print_particles()
#             print('chosen_traj'), print(chosen_trajectory)
#             print("Contents of chosen_trajectory:")
#             for i in range(len(chosen_trajectory)):
#                 print(f"Index {i}: {chosen_trajectory[i]}")

            return chosen_trajectory
