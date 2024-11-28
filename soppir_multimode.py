import os
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse
import os.path

def initialize_particles(position, num_particles, noise_std_dev=0.1, noise_type='gaussian'):
    """
    Initializes particle positions around a given central position with specified noise.

    Parameters:
    - position (array-like): The central position around which particles are initialized.
    - num_particles (int): The number of particles to generate.
    - noise_std_dev (float, optional): Standard deviation of the noise added to the particles. Defaults to 0.1.
    - noise_type (str, optional): Type of noise to apply ('gaussian', 'laplace', or 'zero'). Defaults to 'gaussian'.

    Returns:
    - particles (numpy.ndarray): An array of initialized particle positions.

    This function generates `num_particles` positions around the given `position` by adding noise according to the specified `noise_type`. The positions are wrapped to stay within the range [-0.5, 0.5).
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_std_dev, size=(num_particles, 3))
    elif noise_type == 'laplace':
        b = noise_std_dev / np.sqrt(2)
        noise = np.random.laplace(0, b, size=(num_particles, 3))
    elif noise_type == 'zero':
        noise = np.zeros((num_particles, 3))
    else:
        raise ValueError("Unsupported noise type: choose 'gaussian', 'laplace', or 'zero'")
    
    particles = position + noise
    particles = np.mod(particles + 0.5, 1.0) - 0.5  # Ensure particles stay within [-0.5, 0.5)
    return particles

def predict_particles(particles, velocity, delta_t, noise_std_dev, noise_type='gaussian'):
    """
    Predicts the next positions of particles based on velocity, time delta, and noise.

    Parameters:
    - particles (numpy.ndarray): Current positions of the particles.
    - velocity (array-like): Velocity vector to apply to the particles.
    - delta_t (float): Time step over which to predict the new positions.
    - noise_std_dev (float): Standard deviation of the noise to add.
    - noise_type (str, optional): Type of noise to apply ('gaussian', 'laplace', or 'zero'). Defaults to 'gaussian'.

    Returns:
    - particles (numpy.ndarray): Updated particle positions after prediction.

    This function updates particle positions by applying the motion model: particles move according to the given `velocity` and `delta_t`, with added noise scaled appropriately. The positions are wrapped to stay within the range [-0.5, 0.5).
    """
    scaled_noise_std_dev = noise_std_dev * np.sqrt(delta_t)
    if noise_type == 'gaussian':
        noise = np.random.normal(0, scaled_noise_std_dev, size=particles.shape)
    elif noise_type == 'laplace':
        b = scaled_noise_std_dev / np.sqrt(2)
        noise = np.random.laplace(0, b, size=particles.shape)
    elif noise_type == 'zero':
        noise = np.zeros_like(particles)
    else:
        raise ValueError("Unsupported noise type: choose 'gaussian', 'laplace', or 'zero'")
    
    particles += velocity * delta_t + noise
    particles = np.mod(particles + 0.5, 1.0) - 0.5  # Ensure particles stay within [-0.5, 0.5)
    return particles

def update_weights(particles, actual_position, noise_std_dev=0.01):
    """
    Updates particle weights based on their distance to the actual position.

    Parameters:
    - particles (numpy.ndarray): Current positions of the particles.
    - actual_position (array-like): The observed position to compare against.
    - noise_std_dev (float, optional): Standard deviation used in the weighting function. Defaults to 0.01.

    Returns:
    - weights (numpy.ndarray): Updated and normalized weights for the particles.

    This function computes the weights of the particles based on the Gaussian likelihood of their distance from the `actual_position`. The weights are normalized to sum to one.
    """
    distances = np.linalg.norm(particles - actual_position, axis=1)
    weights = np.exp(-distances**2 / (2 * noise_std_dev**2))
    weights += 1e-300  # To avoid division by zero
    weights /= np.sum(weights)
    return weights

def residual_resample(weights):
    """
    Performs residual resampling of particles based on their weights.

    Parameters:
    - weights (numpy.ndarray): The weights of the particles.

    Returns:
    - indexes (numpy.ndarray): Array of indices indicating which particles are selected.

    This function implements residual resampling, which first takes the integer part of N*w[i], then distributes the residual proportionally to the fractional parts.
    """
    N = len(weights)
    indexes = np.zeros(N, dtype=int)
    
    # take int(N*w) copies of each weight
    weights = np.asarray(weights)
    num_copies = (N * weights).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):  # make num_copies[i] copies
            indexes[k] = i
            k += 1
    
    # Use multinomial resampling on the residual to fill up the rest
    residual = weights - num_copies     # get fractional part
    residual_sum = residual.sum()
    if residual_sum > 0 and k < N:
        residual /= residual_sum        # normalize
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.0        # ensures sum is exactly one
        random_values = np.random.random(N - k)
        indexes[k:N] = np.searchsorted(cumulative_sum, random_values)
    elif k < N:
        # If residual_sum is zero, fill remaining indices randomly
        indexes[k:N] = np.random.choice(N, N - k)
    return indexes

def resample_particles(particles, weights, resampling_method='random'):
    """
    Resamples particles based on their weights using the specified method.

    Parameters:
    - particles (numpy.ndarray): Current positions of the particles.
    - weights (numpy.ndarray): The weights of the particles.
    - resampling_method (str, optional): Method to use for resampling ('random' or 'residual'). Defaults to 'random'.

    Returns:
    - resampled_particles (numpy.ndarray): Resampled particle positions.

    This function resamples the particles according to their weights using the specified resampling method, either 'random' (multinomial) or 'residual'.
    """
    if resampling_method == 'random':
        indices = np.random.choice(len(particles), size=len(particles), p=weights)
    elif resampling_method == 'residual':
        indices = residual_resample(weights)
    else:
        raise ValueError("Unsupported resampling method: choose 'random' or 'residual'")
    return particles[indices]

def sequential_update(particles, velocities, actual_positions, delta_t, noise_std_dev=0.01, noise_type='gaussian', resampling_method='random'):
    """
    Performs sequential updates of particles over all time steps.

    Parameters:
    - particles (numpy.ndarray): Initial positions of the particles.
    - velocities (numpy.ndarray): Array of velocities for each time step.
    - actual_positions (numpy.ndarray): Observed positions at each time step.
    - delta_t (numpy.ndarray): Array of time deltas between steps.
    - noise_std_dev (float, optional): Standard deviation of the noise. Defaults to 0.01.
    - noise_type (str, optional): Type of noise to apply ('gaussian', 'laplace', or 'zero'). Defaults to 'gaussian'.
    - resampling_method (str, optional): Resampling method to use ('random' or 'residual'). Defaults to 'random'.

    Returns:
    - predictions (numpy.ndarray): Array of predicted positions at each time step.

    This function updates the particles through all time steps by predicting, weighting, and resampling at each step, and collects the mean position as the prediction.
    """
    num_steps = len(velocities)
    predictions = []
    for step in range(num_steps):
        particles = predict_particles(particles, velocities[step], delta_t[step], noise_std_dev, noise_type)
        weights = update_weights(particles, actual_positions[step], noise_std_dev)
        predicted_position = np.mean(particles, axis=0)
        predictions.append(predicted_position)
        particles = resample_particles(particles, weights, resampling_method=resampling_method)
    return np.array(predictions)

def process_particles(batch_indices, initial_positions, velocities, actual_positions, delta_t, num_particles_filter, noise_std_dev, noise_type, resampling_method, temp_file_name):
    """
    Processes a batch of particles and writes predicted positions to a temporary file.

    Parameters:
    - batch_indices (numpy.ndarray): Indices of the particles in the batch.
    - initial_positions (numpy.ndarray): Initial positions of the particles in the batch.
    - velocities (numpy.ndarray): Velocities of the particles over time.
    - actual_positions (numpy.ndarray): Observed positions over time.
    - delta_t (numpy.ndarray): Time deltas between observations.
    - num_particles_filter (int): Number of particles to use in the algorithm.
    - noise_std_dev (float): Standard deviation of the noise.
    - noise_type (str): Type of noise to use ('gaussian', 'laplace', or 'zero').
    - resampling_method (str): Resampling method ('random' or 'residual').
    - temp_file_name (str): Name of the temporary file to write the results.

    Returns:
    - None

    This function runs the SOPPIR for each particle in the batch and writes the predicted positions to a temporary HDF5 file.
    """
    num_particles = len(initial_positions)
    num_steps = velocities.shape[1]
    predicted_positions = np.zeros((num_particles, num_steps, 3), dtype=np.float32)
    
    for i in range(num_particles):
        particles = initialize_particles(initial_positions[i], num_particles_filter, noise_std_dev=noise_std_dev, noise_type=noise_type)
        predicted_positions[i] = sequential_update(
            particles,
            velocities[i],
            actual_positions[i],
            delta_t,
            noise_std_dev=noise_std_dev,
            noise_type=noise_type,
            resampling_method=resampling_method
        )
    
    # Write predicted positions to a temporary file
    with h5py.File(temp_file_name, 'w') as temp_h5f:
        temp_h5f.create_dataset('batch_indices', data=batch_indices)
        temp_h5f.create_dataset('predicted_positions', data=predicted_positions)

def process_particles_wrapper(args):
    """
    Wrapper function for multiprocessing to unpack arguments.

    Parameters:
    - args (tuple): Arguments to pass to `process_particles`.

    Returns:
    - None

    This function unpacks the arguments and calls `process_particles`. It is used to facilitate multiprocessing.
    """
    (batch_indices, initial_positions, velocities, actual_positions, delta_t,
     num_particles_filter, noise_std_dev, noise_type, resampling_method, temp_file_name) = args
    process_particles(
        batch_indices,
        initial_positions,
        velocities,
        actual_positions,
        delta_t,
        num_particles_filter,
        noise_std_dev,
        noise_type,
        resampling_method,
        temp_file_name
    )

def merge_temp_files_into_h5(chunks, num_particles, num_steps, ids, times, iterations, output_file_name):
    """
    Merges temporary files into the final HDF5 output file.

    Parameters:
    - chunks (list): List of argument tuples used in processing the chunks.
    - num_particles (int): Total number of particles processed.
    - num_steps (int): Total number of time steps.
    - ids (numpy.ndarray): IDs of the particles.
    - times (numpy.ndarray): Times at each time step.
    - iterations (numpy.ndarray): Iteration numbers at each time step.
    - output_file_name (str): Name of the output HDF5 file.

    Returns:
    - None

    This function reads the predicted positions from temporary files created during processing and writes them into the final output HDF5 file. It also deletes the temporary files afterward.
    """
    with h5py.File(output_file_name, 'w') as h5f:
        h5f.create_dataset('particle_ids', data=ids)
    
        # Iterate over each time step
        for step in tqdm(range(num_steps), desc='Merging Steps'):
            step_group = h5f.create_group(f"Step#{step}")
            # Store 'time' and 'iteration' as arrays to match input file structure
            step_group.attrs['time'] = np.array([times[step + 1]])
            step_group.attrs['iteration'] = np.array([iterations[step + 1]])
    
            # Create datasets for positions
            x_dataset = step_group.create_dataset('x', shape=(num_particles,), dtype='float32')
            y_dataset = step_group.create_dataset('y', shape=(num_particles,), dtype='float32')
            z_dataset = step_group.create_dataset('z', shape=(num_particles,), dtype='float32')
    
            # Read predicted positions from temporary files
            for args in chunks:
                temp_file_name = args[-1]
                with h5py.File(temp_file_name, 'r') as temp_h5f:
                    batch_indices = temp_h5f['batch_indices'][:]
                    predicted_positions = temp_h5f['predicted_positions'][:, step, :]  # Positions at current step
    
                    x_dataset[batch_indices] = predicted_positions[:, 0]
                    y_dataset[batch_indices] = predicted_positions[:, 1]
                    z_dataset[batch_indices] = predicted_positions[:, 2]
    
    # Delete temporary files
    for args in chunks:
        temp_file_name = args[-1]
        try:
            os.remove(temp_file_name)
        except OSError as e:
            print(f"Error deleting temporary file {temp_file_name}: {e}")

def load_data(file_name, num_particles, start_step, end_step):
    """
    Loads data from the input HDF5 file.

    Parameters:
    - file_name (str): Name of the input HDF5 file.
    - num_particles (int): Number of particles to load.
    - start_step (int): Starting time step index.
    - end_step (int): Ending time step index.

    Returns:
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
    
        initial_positions = np.zeros((num_particles, 3), dtype='float32')
        velocities = np.zeros((num_particles, num_steps, 3), dtype='float32')
        actual_positions = np.zeros((num_particles, num_steps, 3), dtype='float32')
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

def main(file_name, resampling_method, start_step, end_step, num_particles_filter, noise_type,
         num_particles=125000, chunk_size=1000, noise_std_dev=0.1):
    """
    Main function to execute the SOPPIR and write predicted positions to an HDF5 file.

    Parameters:
    - file_name (str): Name of the input HDF5 file.
    - resampling_method (str): Resampling method to use ('random' or 'residual').
    - start_step (int): Starting time step index.
    - end_step (int): Ending time step index.
    - num_particles_filter (int): Number of particles in the algorithm.
    - noise_type (str): Type of noise to use ('gaussian', 'laplace', or 'zero').
    - num_particles (int, optional): Number of particles to process. Defaults to 125000.
    - chunk_size (int, optional): Size of chunks for processing particles. Defaults to 1000.
    - noise_std_dev (float, optional): Standard deviation of the noise. Defaults to 0.1.

    Returns:
    - None

    This function orchestrates the entire SOPPIR algorithm process: it loads the data, processes the particles in chunks using multiprocessing, and merges the results into the final output file.
    """
    initial_positions, velocities, actual_positions, ids, times, delta_t, iterations = load_data(
        file_name, num_particles, start_step, end_step)
    
    num_steps = velocities.shape[1]
    
    # Prepare output file name
    file_base_name = os.path.splitext(os.path.basename(file_name))[0]
    output_file_name = f"predicted_positions_{file_base_name}_{num_particles_filter}_{noise_type}_{resampling_method}_cpu.h5"
    
    # Prepare chunks with batch indices and temporary file names
    chunks = []
    for idx, i in enumerate(range(0, num_particles, chunk_size)):
        chunk_start = i
        chunk_end = min(i + chunk_size, num_particles)
        batch_indices = np.arange(chunk_start, chunk_end)
        temp_file_name = f'temp_predicted_positions_{idx}.h5'
    
        args = (
            batch_indices,
            initial_positions[chunk_start:chunk_end],
            velocities[chunk_start:chunk_end],
            actual_positions[chunk_start:chunk_end],
            delta_t,
            num_particles_filter,
            noise_std_dev,
            noise_type,
            resampling_method,
            temp_file_name
        )
        chunks.append(args)
    
    # Process chunks in parallel
    with Pool(processes=min(cpu_count(), 20)) as pool:
        list(tqdm(pool.imap_unordered(process_particles_wrapper, chunks), total=len(chunks), desc='Processing Batches'))
    
    # Merge temporary files into the final HDF5 file
    merge_temp_files_into_h5(chunks, num_particles, num_steps, ids, times, iterations, output_file_name)
    
    print(f"Predicted positions have been successfully written to '{output_file_name}'.")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SOPPIR Script')
    parser.add_argument('file_name', type=str, help='Input HDF5 file name')
    parser.add_argument('resampling_method', type=str, choices=['random', 'residual'], help='Resampling method (random or residual)')
    parser.add_argument('start_step', type=int, help='Starting time step')
    parser.add_argument('end_step', type=int, help='Ending time step')
    parser.add_argument('num_particles_filter', type=int, help='Number of particles in the algorithm')
    parser.add_argument('noise_type', type=str, choices=['gaussian', 'laplace', 'zero'], help='Type of noise (gaussian, laplace, or zero)')
    parser.add_argument('--num_particles', type=int, default=125000, help='Number of particles to process')
    parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size for processing particles')
    parser.add_argument('--noise_std_dev', type=float, default=0.1, help='Standard deviation of noise')
    
    args = parser.parse_args()
    
    main(args.file_name, args.resampling_method, args.start_step, args.end_step, args.num_particles_filter,
         args.noise_type, num_particles=args.num_particles, chunk_size=args.chunk_size, noise_std_dev=args.noise_std_dev)

