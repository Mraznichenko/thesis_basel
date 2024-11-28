import os
import h5py
import numpy as np
from tqdm import tqdm
import argparse
import os.path
import math
from numba import cuda

# Set the number of threads per block (2D)
THREADS_PER_BLOCK = (32, 32)

@cuda.jit(device=True)
def xorshift32(s):
    """
    Performs a 32-bit XORSHIFT random number generation step.

    Parameters:
    - s (int): The seed or current state of the random number generator.

    Returns:
    - int: Updated seed after applying the XORSHIFT algorithm.

    This function implements the XORSHIFT algorithm to produce pseudorandom numbers, suitable for use within CUDA kernels.
    """
    s ^= (s << 13) & 0xFFFFFFFF
    s ^= (s >> 17)
    s ^= (s << 5) & 0xFFFFFFFF
    return s & 0xFFFFFFFF

@cuda.jit(device=True)
def rand_uniform(s):
    """
    Generates a uniform random number in [0, 1) and updates the seed.

    Parameters:
    - s (int): Current seed for the random number generator.

    Returns:
    - tuple: A tuple containing:
        - float: A uniform random number in the range [0, 1).
        - int: Updated seed after random number generation.

    This function uses the XORSHIFT algorithm to generate a pseudorandom number and normalizes it to [0, 1).
    """
    s = xorshift32(s)
    return s / 4294967296.0, s  # Normalize to [0, 1) and return updated seed

@cuda.jit(device=True)
def randn(s):
    """
    Generates a standard normally distributed random number using the Box-Muller transform.

    Parameters:
    - s (int): Current seed for the random number generator.

    Returns:
    - tuple: A tuple containing:
        - float: A normally distributed random number (mean=0, std=1).
        - int: Updated seed after random number generation.

    This function generates a random number following a standard normal distribution using the Box-Muller transform.
    """
    u1, s = rand_uniform(s)
    u2, s = rand_uniform(s)
    z = math.sqrt(-2 * math.log(u1 + 1e-10)) * math.cos(2 * math.pi * u2)
    return z, s

@cuda.jit(device=True)
def rand_laplace(s, scale):
    """
    Generates a Laplace-distributed random number.

    Parameters:
    - s (int): Current seed for the random number generator.
    - scale (float): Scale parameter (diversity) of the Laplace distribution.

    Returns:
    - tuple: A tuple containing:
        - float: A random number following the Laplace distribution.
        - int: Updated seed after random number generation.

    This function generates a random number from a Laplace distribution centered at zero with the specified `scale`.
    """
    u, s = rand_uniform(s)
    u = u - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u) + 1e-10), s

@cuda.jit
def initialize_particles_kernel(initial_positions, particles, noise_std_dev, noise_type, seeds):
    """
    Initializes particles around initial positions with specified noise on the GPU.

    Parameters:
    - initial_positions (numpy.ndarray): Array of initial positions for each particle.
    - particles (numpy.ndarray): Array to store the initialized particle positions.
    - noise_std_dev (float): Standard deviation of the noise to add.
    - noise_type (int): Type of noise (0 for Gaussian, 1 for Laplace, 2 for zero noise).
    - seeds (numpy.ndarray): Seed array for random number generation per particle.

    Returns:
    - None

    This kernel function initializes particles by adding noise to the initial positions. It ensures particles stay within the range [-0.5, 0.5).
    """
    x, y = cuda.grid(2)
    num_particles = initial_positions.shape[0]
    num_particles_filter = particles.shape[1]
    if x < num_particles and y < num_particles_filter:
        seed = seeds[x, y]

        for d in range(3):
            if noise_type == 0:  # Gaussian
                noise, seed = randn(seed)
                noise *= noise_std_dev
            elif noise_type == 1:  # Laplace
                b = noise_std_dev / math.sqrt(2)
                noise, seed = rand_laplace(seed, b)
            elif noise_type == 2:  # Zero noise
                noise = 0.0
            else:
                noise = 0.0  # Default to zero noise
            pos = initial_positions[x, d] + noise
            # Ensure particles stay within [-0.5, 0.5)
            particles[x, y, d] = (pos + 0.5) % 1.0 - 0.5

        seeds[x, y] = seed

@cuda.jit
def predict_particles_kernel(particles, velocities, delta_t, noise_std_dev, noise_type, seeds):
    """
    Predicts the next positions of particles based on velocities and noise on the GPU.

    Parameters:
    - particles (numpy.ndarray): Current positions of the particles.
    - velocities (numpy.ndarray): Velocities of the particles.
    - delta_t (float): Time step over which to predict the new positions.
    - noise_std_dev (float): Standard deviation of the noise to add.
    - noise_type (int): Type of noise (0 for Gaussian, 1 for Laplace, 2 for zero noise).
    - seeds (numpy.ndarray): Seed array for random number generation per particle.

    Returns:
    - None

    This kernel function updates particle positions using the motion model and noise, ensuring they stay within [-0.5, 0.5).
    """
    x, y = cuda.grid(2)
    num_particles = velocities.shape[0]
    num_particles_filter = particles.shape[1]
    if x < num_particles and y < num_particles_filter:
        seed = seeds[x, y]

        for d in range(3):
            if noise_type == 0:  # Gaussian
                noise, seed = randn(seed)
                noise *= noise_std_dev * math.sqrt(delta_t)
            elif noise_type == 1:  # Laplace
                b = (noise_std_dev * math.sqrt(delta_t)) / math.sqrt(2)
                noise, seed = rand_laplace(seed, b)
            elif noise_type == 2:  # Zero noise
                noise = 0.0
            else:
                noise = 0.0  # Default to zero noise
            pos = particles[x, y, d] + velocities[x, d] * delta_t + noise
            particles[x, y, d] = (pos + 0.5) % 1.0 - 0.5

        seeds[x, y] = seed

@cuda.jit
def update_weights_kernel(particles, actual_positions, weights, noise_std_dev):
    """
    Updates the weights of particles based on their proximity to the actual positions.

    Parameters:
    - particles (numpy.ndarray): Current positions of the particles.
    - actual_positions (numpy.ndarray): Observed positions to compare against.
    - weights (numpy.ndarray): Array to store the updated weights.
    - noise_std_dev (float): Standard deviation used in the weighting function.

    Returns:
    - None

    This kernel function computes the weights of particles using a Gaussian likelihood based on their distance from the observed positions.
    """
    x, y = cuda.grid(2)
    num_particles = actual_positions.shape[0]
    num_particles_filter = particles.shape[1]
    if x < num_particles and y < num_particles_filter:
        dist_sq = 0.0
        for d in range(3):
            diff = particles[x, y, d] - actual_positions[x, d]
            dist_sq += diff * diff

        weights[x, y] = math.exp(-dist_sq / (2 * noise_std_dev ** 2)) + 1e-300  # Avoid zero

@cuda.jit
def normalize_weights_kernel(weights):
    """
    Normalizes the weights of particles so that they sum to one.

    Parameters:
    - weights (numpy.ndarray): Array of weights to be normalized.

    Returns:
    - None

    This kernel function normalizes the weights for each particle set by dividing by the total sum. If the total sum is zero, it assigns equal weights.
    """
    x = cuda.grid(1)
    num_particles = weights.shape[0]
    num_particles_filter = weights.shape[1]
    if x < num_particles:
        weight_sum = 0.0
        for y in range(num_particles_filter):
            weight_sum += weights[x, y]
        if weight_sum > 0:
            for y in range(num_particles_filter):
                weights[x, y] /= weight_sum
        else:
            inv_num = 1.0 / num_particles_filter
            for y in range(num_particles_filter):
                weights[x, y] = inv_num

@cuda.jit
def resample_particles_kernel(particles, weights, seeds, particles_out, resampling_method):
    """
    Resamples particles based on their weights using the specified method.

    Parameters:
    - particles (numpy.ndarray): Current positions of the particles.
    - weights (numpy.ndarray): Weights of the particles.
    - seeds (numpy.ndarray): Seed array for random number generation per particle.
    - particles_out (numpy.ndarray): Array to store resampled particle positions.
    - resampling_method (int): Resampling method (0 for random/multinomial).

    Returns:
    - None

    This kernel function resamples particles according to their weights using the multinomial (random) resampling method.
    """
    x, y = cuda.grid(2)
    num_particles = particles.shape[0]
    num_particles_filter = particles.shape[1]
    if x < num_particles and y < num_particles_filter:
        seed = seeds[x, y]

        # Multinomial (random) resampling
        rnd, seed = rand_uniform(seed)
        cumulative_sum = 0.0
        selected_idx = -1
        for i in range(num_particles_filter):
            cumulative_sum += weights[x, i]
            if rnd <= cumulative_sum:
                selected_idx = i
                break
        if selected_idx == -1:
            selected_idx = num_particles_filter - 1  # Handle edge case

        for d in range(3):
            particles_out[x, y, d] = particles[x, selected_idx, d]

        seeds[x, y] = seed

@cuda.jit
def compute_predictions_kernel(particles, predictions):
    """
    Computes the mean position of particles to form the prediction.

    Parameters:
    - particles (numpy.ndarray): Positions of the particles.
    - predictions (numpy.ndarray): Array to store the computed mean positions.

    Returns:
    - None

    This kernel function calculates the average position of particles for each particle set to generate the predicted positions.
    """
    x = cuda.grid(1)
    num_particles = particles.shape[0]
    num_particles_filter = particles.shape[1]
    if x < num_particles:
        sum_pos = cuda.local.array(3, dtype=particles.dtype)
        sum_pos[0] = 0.0
        sum_pos[1] = 0.0
        sum_pos[2] = 0.0
        for y in range(num_particles_filter):
            for d in range(3):
                sum_pos[d] += particles[x, y, d]
        for d in range(3):
            predictions[x, d] = sum_pos[d] / num_particles_filter

def process_particles(initial_positions, velocities, actual_positions, delta_t, num_particles_filter, noise_std_dev, noise_type, resampling_method):
    """
    Processes particles using the GPU to perform SOPPIR algorithm.

    Parameters:
    - initial_positions (numpy.ndarray): Initial positions of the particles.
    - velocities (numpy.ndarray): Velocities of the particles over time.
    - actual_positions (numpy.ndarray): Observed positions over time.
    - delta_t (numpy.ndarray): Time deltas between observations.
    - num_particles_filter (int): Number of particles in the algorithm.
    - noise_std_dev (float): Standard deviation of the noise.
    - noise_type (str): Type of noise ('gaussian', 'laplace', or 'zero').
    - resampling_method (str): Resampling method ('random' or 'residual').

    Returns:
    - predictions (numpy.ndarray): Predicted positions over time for each particle.

    This function orchestrates the SOPPIR process on the GPU, initializing particles, predicting their movement, updating weights, resampling, and computing predictions.
    """
    num_particles = initial_positions.shape[0]
    num_steps = velocities.shape[1]

    # Prepare seeds for RNG
    seeds = np.random.randint(0, 4294967295, size=(num_particles, num_particles_filter), dtype=np.uint32)

    # Map noise type string to integer for CUDA kernels
    noise_type_map = {'gaussian': 0, 'laplace': 1, 'zero': 2}
    noise_type_int = noise_type_map.get(noise_type, 0)

    # Map resampling method string to integer
    resampling_method_map = {'random': 0, 'residual': 1}
    resampling_method_int = resampling_method_map.get(resampling_method, 0)

    # Allocate device memory
    particles = np.zeros((num_particles, num_particles_filter, 3), dtype=np.float32)
    particles_device = cuda.to_device(particles)
    velocities_device = cuda.to_device(velocities)
    initial_positions_device = cuda.to_device(initial_positions)
    actual_positions_device = cuda.to_device(actual_positions)
    seeds_device = cuda.to_device(seeds)
    weights_device = cuda.device_array(shape=(num_particles, num_particles_filter), dtype=np.float32)
    predictions = np.zeros((num_particles, num_steps, 3), dtype=np.float32)
    predictions_device = cuda.to_device(predictions)

    threads_per_block = THREADS_PER_BLOCK
    blocks_per_grid_x = (num_particles + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (num_particles_filter + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Initialize particles
    initialize_particles_kernel[blocks_per_grid, threads_per_block](
        initial_positions_device, particles_device, noise_std_dev, noise_type_int, seeds_device)
    cuda.synchronize()

    for step in tqdm(range(num_steps), desc='Processing Steps'):
        # Predict particles
        delta_t_step = delta_t[step]
        predict_particles_kernel[blocks_per_grid, threads_per_block](
            particles_device, velocities_device[:, step, :], delta_t_step, noise_std_dev, noise_type_int, seeds_device)
        cuda.synchronize()

        # Update weights
        update_weights_kernel[blocks_per_grid, threads_per_block](
            particles_device, actual_positions_device[:, step, :], weights_device, noise_std_dev)
        cuda.synchronize()

        # Normalize weights
        normalize_weights_kernel[(blocks_per_grid_x * threads_per_block[0],), (threads_per_block[0],)](
            weights_device)
        cuda.synchronize()

        # Compute predictions
        compute_predictions_kernel[(blocks_per_grid_x * threads_per_block[0],), (threads_per_block[0],)](
            particles_device, predictions_device[:, step, :])
        cuda.synchronize()

        # Resample particles
        particles_out_device = cuda.device_array_like(particles_device)
        resample_particles_kernel[blocks_per_grid, threads_per_block](
            particles_device, weights_device, seeds_device, particles_out_device, resampling_method_int)
        cuda.synchronize()

        # Swap particles
        particles_device, particles_out_device = particles_out_device, particles_device

    # Copy predictions back to host
    predictions = predictions_device.copy_to_host()

    return predictions

def main(file_name, resampling_method, start_step, end_step, num_particles_filter, noise_type,
         num_particles=125000, noise_std_dev=0.1):
    """
    Main function to execute the SOPPIR on the GPU and write predicted positions to an HDF5 file.

    Parameters:
    - file_name (str): Name of the input HDF5 file.
    - resampling_method (str): Resampling method to use ('random' or 'residual').
    - start_step (int): Starting time step index.
    - end_step (int): Ending time step index.
    - num_particles_filter (int): Number of particles in the alogorithm.
    - noise_type (str): Type of noise ('gaussian', 'laplace', or 'zero').
    - num_particles (int, optional): Number of particles to process. Defaults to 125000.
    - noise_std_dev (float, optional): Standard deviation of the noise. Defaults to 0.1.

    Returns:
    - None

    This function orchestrates the loading of data, processing of particles using GPU acceleration, and saving of predicted positions to an output HDF5 file.
    """
    initial_positions, velocities, actual_positions, ids, times, delta_t, iterations = load_data(
        file_name, num_particles, start_step, end_step)

    num_steps = velocities.shape[1]

    # Prepare output file name
    file_base_name = os.path.splitext(os.path.basename(file_name))[0]
    output_file_name = f"predicted_positions_{file_base_name}_{num_particles_filter}_{noise_type}_{resampling_method}_gpu.h5"

    # Process particles using GPU
    predictions = process_particles(
        initial_positions,
        velocities,
        actual_positions,
        delta_t,
        num_particles_filter,
        noise_std_dev,
        noise_type,
        resampling_method
    )

    # Save predictions to HDF5 file
    with h5py.File(output_file_name, 'w') as h5f:
        h5f.create_dataset('particle_ids', data=ids)

        # Iterate over each time step
        for step in tqdm(range(num_steps), desc='Saving Steps'):
            step_group = h5f.create_group(f"Step#{step}")
            # Store 'time' and 'iteration' as arrays to match input file structure
            step_group.attrs['time'] = np.array([times[step + 1]])
            step_group.attrs['iteration'] = np.array([iterations[step + 1]])

            # Create datasets for positions
            x_dataset = step_group.create_dataset('x', data=predictions[:, step, 0])
            y_dataset = step_group.create_dataset('y', data=predictions[:, step, 1])
            z_dataset = step_group.create_dataset('z', data=predictions[:, step, 2])

    print(f"Predicted positions have been successfully written to '{output_file_name}'.")

def load_data(file_name, num_particles, start_step, end_step):
    """
    Loads data from the input HDF5 file.

    Parameters:
    - file_name (str): Name of the input HDF5 file.
    - num_particles (int): Number of particles to load.
    - start_step (int): Starting time step index.
    - end_step (int): Ending time step index.

    Returns:
    - tuple: A tuple containing:
        - initial_positions (numpy.ndarray): Initial positions of the particles.
        - velocities (numpy.ndarray): Velocities of the particles over time.
        - actual_positions (numpy.ndarray): Observed positions over time.
        - ids (numpy.ndarray): IDs of the particles.
        - times (numpy.ndarray): Times at each time step.
        - delta_t (numpy.ndarray): Time deltas between time steps.
        - iterations (numpy.ndarray): Iteration numbers at each time step.

    This function reads particle positions, velocities, times, and iteration numbers from the specified HDF5 file for the given range of time steps.
    """
    with h5py.File(file_name, 'r') as f:
        num_steps = end_step - start_step
        ids = f[f'Step#{start_step}']['id'][:num_particles]

        initial_positions = np.zeros((num_particles, 3), dtype=np.float32)
        velocities = np.zeros((num_particles, num_steps, 3), dtype=np.float32)
        actual_positions = np.zeros((num_particles, num_steps, 3), dtype=np.float32)
        times = np.zeros(num_steps + 1, dtype='float64')
        iterations = np.zeros(num_steps + 1, dtype=int)

        # Read initial positions, time, and iteration
        h5step = f[f'Step#{start_step}']
        times[0] = h5step.attrs['time'][0]
        iterations[0] = h5step.attrs['iteration'][0]
        initial_positions[:, 0] = h5step['x'][:num_particles]
        initial_positions[:, 1] = h5step['y'][:num_particles]
        initial_positions[:, 2] = h5step['z'][:num_particles]

        # Read velocities, times, and iterations
        for i, step in enumerate(range(start_step, end_step)):
            h5step = f[f'Step#{step}']
            times[i + 1] = h5step.attrs['time'][0]
            iterations[i + 1] = h5step.attrs['iteration'][0]

            velocities[:, i, 0] = h5step['vx'][:num_particles]
            velocities[:, i, 1] = h5step['vy'][:num_particles]
            velocities[:, i, 2] = h5step['vz'][:num_particles]

        # Read actual positions
        for i, step in enumerate(range(start_step + 1, end_step + 1)):
            h5step = f[f'Step#{step}']
            actual_positions[:, i - 1, 0] = h5step['x'][:num_particles]
            actual_positions[:, i - 1, 1] = h5step['y'][:num_particles]
            actual_positions[:, i - 1, 2] = h5step['z'][:num_particles]

        delta_t = np.diff(times)

    return initial_positions, velocities, actual_positions, ids, times, delta_t, iterations

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SOPPIR GPU Script')
    parser.add_argument('file_name', type=str, help='Input HDF5 file name')
    parser.add_argument('resampling_method', type=str, choices=['random', 'residual'], help='Resampling method (random or residual)')
    parser.add_argument('start_step', type=int, help='Starting time step')
    parser.add_argument('end_step', type=int, help='Ending time step')
    parser.add_argument('num_particles_filter', type=int, help='Number of particles in the algorithm')
    parser.add_argument('noise_type', type=str, choices=['gaussian', 'laplace', 'zero'], help='Type of noise (gaussian, laplace, or zero)')
    parser.add_argument('--num_particles', type=int, default=125000, help='Number of particles to process')
    parser.add_argument('--noise_std_dev', type=float, default=0.1, help='Standard deviation of noise')

    args = parser.parse_args()

    main(args.file_name, args.resampling_method, args.start_step, args.end_step, args.num_particles_filter,
         args.noise_type, num_particles=args.num_particles, noise_std_dev=args.noise_std_dev)


