 
print('importing libraries')
import json
import numpy as np
import os
import datetime
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.decomposition import PCA
import astropy.io.fits as pyfits
import sys
import gc
from multiprocessing import Pool
from itertools import chain
import multiprocessing
import matplotlib.lines as mlines
import time
start_time = time.time()
from matplotlib.colors import LogNorm
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import h5py
import threading
from concurrent.futures import ThreadPoolExecutor
"""from tensorflow.keras import mixed_precision

# Set the global policy
mixed_precision.set_global_policy('mixed_float32')
"""
tf.config.threading.set_inter_op_parallelism_threads(6)  # For coordinating independent operations
tf.config.threading.set_intra_op_parallelism_threads(6)  # For speeding up individual operations


logs = "tf_logs/"

print("Available GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
print('Imports Complete')
print('number of cores available:' + str(multiprocessing.cpu_count()))
print('Initializing Fcns and Classes')

print("Available devices:")
print(tf.config.list_physical_devices('GPU'))

# Check if a GPU is available and TensorFlow is using it
if tf.test.is_gpu_available():
    print("\n\nTensorFlow will run on GPU.\n\n")
else:
    print("\n\nTensorFlow will run on CPU.\n\n")


executor = ThreadPoolExecutor(max_workers=6)  # Adjust as needed



class FITSDataExtractor:
    """
    Class to handle the extraction and processing of data from FITS files.
    """

    def __init__(self, file_directory):
        """
        Initialize the FITSDataExtractor.

        Args:
        file_directory (str): Directory where FITS files are located.
        """
        self.file_directory = file_directory
        self.min_loglam = []
        self.max_loglam = []
        self.loglam = []
        self.interp_spacing = []
    def check_loglam(self, filename):
        """
        Extracts and processes data from a single FITS file.

        Args:
        filename (str): Name of the FITS file to process.

        Returns:
        Tuple: Processed data elements (flux, redshift, class, filepath)
        """
        if filename.endswith('.fits') or filename.endswith('1'):
            try:
                with pyfits.open(self.file_directory + filename) as fit_file:
                    # Try different naming conventions
                    try:
                        coadd = fit_file['COADD']
                        spall = fit_file['SPALL']
                    except KeyError:
                        coadd = fit_file['COADD']
                        spall = fit_file['SPECOBJ']

                    # Process data
                    loglam_wavelength_data = np.array(coadd.data['loglam'])
                    redshift_data = spall.data['Z']
                    obs_wavelength_data = 10**loglam_wavelength_data
                    eff_wavelength_data = obs_wavelength_data #/ (1 + redshift_data)
                    avg_spacing = np.mean(np.diff(eff_wavelength_data))
                    return np.array([min(eff_wavelength_data), max(eff_wavelength_data)]), avg_spacing
                
            except Exception as e:
                print(f"Failed with exception: {e} on filename: {filename}")
                return None, None


    def extract_data(self, filename):
        """
        Extracts and processes data from a single FITS file.

        Args:
        filename (str): Name of the FITS file to process.

        Returns:
        Tuple: Processed data elements (flux, redshift, class, filepath)
        """
        if filename.endswith('.fits') or filename.endswith('1'):
            try:
                with pyfits.open(self.file_directory + filename) as fit_file:
                    # Try different naming conventions
                    try:
                        coadd = fit_file['COADD']
                        spall = fit_file['SPALL']

                    except KeyError:
                        coadd = fit_file['COADD']
                        spall = fit_file['SPECOBJ']

                    # Process data
                    
                    #Extract data
                    loglam_wavelength_data = np.array(coadd.data['loglam'])
                    object_class = spall.data['CLASS']
                    redshift_data = spall.data['Z']
                    flux_data = np.array(coadd.data['flux'])
                    ivar_data = np.array(coadd.data['ivar'])

                    #apply min-max normalization to flux
                    min_value = 0
                    max_value = 50
                    
                    # Calculate the min and max values of the data
                    data_min = np.min(flux_data)
                    data_max = np.max(flux_data)
                    # Perform Min-Max normalization
                    flux_data = (flux_data - data_min) / (data_max - data_min) * (max_value - min_value) + min_value
                    
                    #convert to angstroms and deredshift wavelength data
                    obs_wavelength_data = 10**loglam_wavelength_data
                    eff_wavelength_data = obs_wavelength_data 
                    
                    #interpolate data
                    common_wavelengths = np.arange(3000, 9000, 1)#self.min_loglam, self.max_loglam, 1)
                    out = np.interp(common_wavelengths, eff_wavelength_data, flux_data, left=0, right=0)
                    ivar_data = np.interp(common_wavelengths, eff_wavelength_data, ivar_data, left=0, right=0)
                    
                    #calculate SNR for filtering
                    SNR = np.array(out*np.sqrt(ivar_data))
                    SNR = SNR[SNR>0]
                    SNR = np.median(SNR)
                    
                    if SNR < 5:
                        return None, None, None, None, None, None
                    
                    return out, redshift_data, object_class, self.file_directory + filename, ivar_data, SNR
            except Exception as e:
                print(f"Failed with exception: {e} on filename: {filename}")
                return None, None , None  , None, None, None        
                    
        
        
    def parallel_open_files(self):
        """
        Processes multiple FITS files in parallel and returns a DataFrame of the data.
        """
        filenames = [filename for filename in os.listdir(self.file_directory) if filename.endswith('.fits') or filename.endswith('1')][:600000]

        all_flux_data = []
        all_redshift_data = []
        all_class_data = []
        all_filename_data = []
        all_ivar_data = []
        all_snr_data = []
        with Pool() as pool:
            loglam_results = pool.map(self.check_loglam, filenames)
        
        # Unpack results - separating min-max wavelengths and average spacings
        min_max_loglams, avg_spacings = zip(*[(min_max, avg) for min_max, avg in loglam_results if min_max is not None and avg is not None])
        
        # Now, find the overall min and max wavelengths and median of average spacings
        self.min_loglam = min(min_max_loglams, key=lambda x: x[0])[0]
        self.max_loglam = max(min_max_loglams, key=lambda x: x[1])[1]
        self.interp_spacing = np.median(avg_spacings)
        
        with Pool() as pool:
            results = pool.map(self.extract_data, filenames)

        # Process results
        for out, redshift_data, object_class, filepath, ivar_data, SNR in results:
            if out is not None:  # Filter out failed extractions
                all_flux_data.append(out)
                all_redshift_data.append(redshift_data)
                all_class_data.append(object_class)
                all_filename_data.append(filepath)
                all_ivar_data.append(ivar_data)
                all_snr_data.append(SNR)
        # Compile data into DataFrame
        data_matrix = pd.DataFrame({
            'interpol_flux': all_flux_data,
            'ivar': all_ivar_data,
            'Z': all_redshift_data,
            'class': all_class_data,
            'filepath': all_filename_data,
            'snr': all_snr_data,
        })



        return data_matrix


def check_and_empty_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    # Check if the directory is empty
    if not os.listdir(directory_path):
        print(f"Directory '{directory_path}' is already empty.")
        return
    
    # If the directory is not empty, remove all files and subdirectories
    try:
        shutil.rmtree(directory_path)
        os.mkdir(directory_path)  # Recreate the empty directory
        print(f"Directory '{directory_path}' has been emptied.")
    except Exception as e:
        print(f"Error: {e}")


def load_som_model(load_path):
    # Load the configurations
    config_path = os.path.join(load_path, 'som_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Reconstruct the SOM with the saved configurations
    som = SOM(m=config["m"], n=config["n"], dim=config["dim"], alpha=0.0, sigma=0.0, batch_size=256)  # Adjust alpha, sigma, and batch_size as needed
    
    # Load and set the weights
    weights_path = os.path.join(load_path, 'som_weights.npy')
    som.som_weights.assign(np.load(weights_path))
    
    return som
def compute_average(cluster_id, data_matrix):
    # Filter data points belonging to the specified cluster
    cluster_data = data_matrix[data_matrix['som_cluster'] == cluster_id]['Z']
    if not cluster_data.empty:
        return cluster_id, np.average(cluster_data)
    else:
        return cluster_id, 0

def parallel_compute_averages(data_matrix, num_clusters=64*64):
    # Create a list of cluster IDs based on your SOM's structure
    cluster_ids = list(range(num_clusters))  # Adjust based on how you identify clusters
    
    with Pool() as pool:
        results = pool.starmap(compute_average, [(cluster_id, data_matrix) for cluster_id in cluster_ids])

    # Initialize an empty result structure
    averages = {cluster_id: avg for cluster_id, avg in results}
    return averages



def compute_average(cluster_id, z_values):
    if z_values.size > 0:
        return cluster_id, np.average(z_values)
    else:
        return cluster_id, np.nan

def parallel_compute_averages(data_matrix, num_clusters=64*64):
    pool = Pool()
    cluster_ids = range(num_clusters)
    results = pool.starmap(compute_average, [(cid, data_matrix[data_matrix['som_cluster'] == cid]['Z'].values) for cid in cluster_ids])
    pool.close()
    pool.join()
    return dict(results)

def plot_redshift(data_matrix, file_number):
    results = parallel_compute_averages(data_matrix)
    grid = np.full((64, 64), np.nan)
    for cluster_id, avg in results.items():
        i, j = divmod(cluster_id, 64)
        grid[i, j] = avg

    im = plt.imshow(grid, cmap='plasma')
    cbar = plt.colorbar()
    cbar.set_label('Objects per pixel')
    plt.title('Redshift Distribution')
    plt.savefig(f'redshift_maps/redshift_som{file_number}.pdf')
    plt.close()


def plot_checkpoint(som_placement, data_matrix, file_number):
    data_matrix = data_matrix.copy()

    data_matrix['som_cluster'] = som_placement
    clusters = [(i, j) for i in range(64) for j in range(64)]

    # Prepare the arguments for each task
    args = [(i, j) for i, j in clusters]

    plot_redshift(data_matrix, file_number)

    print('plotting quasars')
    qso_clusters = [row for row in restrict_column(data_matrix, 'class', 'QSO')['som_cluster']]
    qso_clusters_init = np.zeros((64, 64))
    for linear_index in qso_clusters:
        i = linear_index // 64  # Convert linear index to 2D grid row
        j = linear_index % 64   # Convert linear index to 2D grid column
        qso_clusters_init[i, j] += 1  # Increment the count for the cluster     


    plt.figure()
    plt.title('Quasar SOM')
    im = plt.imshow(qso_clusters_init, cmap = 'plasma')
    cbar = plt.colorbar()
    cbar.set_label('Objects per pixel')
    plt.savefig(f'qso_maps/qso_som{file_number}.pdf')

    plt.figure()
    plt.title('Quasar SOM')
    im = plt.imshow(qso_clusters_init, cmap = 'plasma', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Objects per pixel')
    plt.savefig(f'qso_log_maps/qso_log_som{file_number}.pdf')
    
    print('plotting galaxies')
    gal_clusters = [row for row in restrict_column(data_matrix, 'class', 'GALAXY')['som_cluster']]
    gal_clusters_init = np.zeros((64, 64))
    for linear_index in gal_clusters:
        i = linear_index // 64  # Convert linear index to 2D grid row
        j = linear_index % 64   # Convert linear index to 2D grid column
        gal_clusters_init[i, j] += 1  # Increment the count for the cluster        
        
    plt.figure()
    plt.title('Galaxy SOM')
    im = plt.imshow(gal_clusters_init, cmap = 'plasma')
    cbar = plt.colorbar()
    cbar.set_label('Objects per pixel')
    plt.savefig(f'gal_maps/gal_som{file_number}.pdf')

    plt.figure()
    plt.title('Galaxy SOM')
    im = plt.imshow(gal_clusters_init, cmap = 'plasma', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Objects per pixel')
    plt.savefig(f'gal_log_maps/gal_log_som{file_number}.pdf')

    print('plotting all')
    all_clusters = som_placement
    all_clusters_init = np.zeros((64, 64))
    for linear_index in all_clusters:
        i = linear_index // 64  # Convert linear index to 2D grid row
        j = linear_index % 64   # Convert linear index to 2D grid column
        all_clusters_init[i, j] += 1  # Increment the count for the cluster        
        
    plt.figure()
    plt.title('all som')
    im = plt.imshow(all_clusters_init, cmap = 'plasma')
    cbar = plt.colorbar()
    cbar.set_label('Objects per pixel')
    plt.savefig(f'all_maps/all_som{file_number}.pdf')

    plt.figure()
    plt.title('all som')
    im = plt.imshow(all_clusters_init, cmap = 'plasma', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Objects per pixel')
    plt.savefig(f'all_log_maps/all_log_som{file_number}.pdf')
    plt.close('all')
    del data_matrix, som_placement, qso_clusters, gal_clusters, all_clusters
    gc.collect()

def plot_in_background(plot_checkpoint_fn, args):
    plot_thread = threading.Thread(target=plot_checkpoint_fn, args=args)
    plot_thread.start()
    return plot_thread

def restrict_column(data_matrix, column, keep):
    out = data_matrix[data_matrix[column] == keep]
    return out

class autoencoder:
    def __init__(self, data, data_matrix, alpha_loss=5e-4, ae_lr=0.001, som_lr=0.001, som_radius=1, som_height=16, som_width=16, alpha=0.001, sigma=1, ae_epochs=50, som_epochs=25, T=1, T_min=0.1, batch_size=128, encoding_dim=100, som_batches = 128):
        """
        Initialize the SOM in TensorFlow.
        """
        self.m = som_width
        self.n = som_height
        self.som_lr = som_lr
        self.data_matrix = data_matrix
        self.ae_lr = ae_lr
        self.som_epochs = som_epochs
        self.ae_epochs = ae_epochs
        self.alpha = alpha
        self.alpha_loss = alpha_loss
        self.sigma = sigma
        # Initialize weights using TensorFlow's random normal function
        self.T = T
        self.T_min = T_min
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.som_radius = som_radius
        self.data = data
        self.som_batches = som_batches


    def decay_fcn(self, epochs, epoch, T):
        T = self.T*(self.T_min/self.T)**(epoch/epochs)
        return T

    def train_and_evaluate_autoencoder(self, output_dir, epochs=200, batch_size=16, patience=25):
        """
        Trains an autoencoder on the provided data, evaluates its training loss, and returns the encoder model.
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        
    
        input_img = Input(shape=(6000,))#data.shape[1],))
    
        # Encoder with BatchNormalization
        encoded = Dense(2048, activation='relu')(input_img)
        encoded = Dropout(0.1)(encoded)

        encoded = Dense(1024, activation='relu')(encoded)
        encoded = Dropout(0.1)(encoded)

        encoded = Dense(512, activation='relu')(encoded)
        encoded = Dropout(0.1)(encoded)

        encoded = Dense(256, activation='relu')(encoded)

        encoded_output = Dense(self.encoding_dim, activation='relu')(encoded)  # This is the encoded representation
    
        # Decoder
        decoded = Dense(256, activation='relu')(encoded_output)

        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dropout(0.1)(decoded)

        decoded = Dense(1024, activation='relu')(decoded)
        decoded = Dropout(0.1)(decoded)

        decoded = Dense(2048, activation='relu')(decoded)
        decoded = Dropout(0.1)(decoded)

        decoded = Dense(6000, activation='relu')(decoded)


        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded_output)  # Model to encode input
        decoder = Model(encoded_output, decoded)
        

        opt = Adam(learning_rate=self.ae_lr) 
        #opt1 = Adam(learning_rate=self.ae_lr) 
        #opt2 = Adam(learning_rate=self.ae_lr) 
        
            
        # SOM parameters
        m, n = self.m, self.n  # Grid size
        epochs_som = self.som_epochs
        alpha = self.som_lr  # Learning rate
        sigma = self.som_radius  # Neighborhood radius
        
        global som
        global total_losses
        total_losses = []
        # Initialize your SOM instance (ensure it's compatible with TensorFlow operations)
        som = SOM(m, n, self.encoding_dim, alpha, sigma, self.som_batches)
        opt = Adam(learning_rate=self.ae_lr) 

        # Prepare the dataset for training (assuming `data` is a NumPy array or a TensorFlow dataset)
        def hdf5_batch_generator(filename, batch_size, total_samples):
            """
            Generator function to yield batches of data from an HDF5 file.
            
            Args:
            filename (str): Path to the HDF5 file.
            batch_size (int): Number of samples per batch.
            total_samples (int): Total number of samples to read from the file.
            """
            with h5py.File(filename, 'r') as f:
                dataset = f['flux']  # Adjust based on your dataset key
                for i in range(0, total_samples, batch_size):
                    end_index = min(i + batch_size, total_samples)
                    data_batch = dataset[i:end_index]
                    if data_batch.shape[0] == 0:  # Check to avoid yielding empty batches
                        break
                    # Ensure data is in the correct format and shape for TensorFlow
                    data_batch = np.array(data_batch, dtype=np.float32)
                    yield data_batch

        # Usage example
        filename = 'all_flux.h5'
        batch_size = self.batch_size  # Example batch size
        total_samples = 100000  # Example total samples, adjust as needed

        dataset = tf.data.Dataset.from_generator(
            lambda: hdf5_batch_generator(filename, batch_size, total_samples),
            output_types=tf.float32,
            output_shapes=(None, 6000)
        ).prefetch(buffer_size=tf.data.AUTOTUNE)
        #dataset = tf.data.Dataset.from_tensor_slices(self.data).cache()
        #dataset = dataset.shuffle(buffer_size=len(self.data)).batch(self.batch_size)

        print('training')

        

        initialized_som = False
        sample_encoded_data = []
        for step, batch_data in enumerate(dataset):
            print(np.shape(sample_encoded_data))
            if step <= 300:
                sample_encoded_data.extend(encoder(batch_data).numpy().reshape(-1, self.encoding_dim))
            else:
                break
        """kmeans = KMeans(n_clusters=self.m * self.n, random_state=0).fit(sample_encoded_data)
        centroids = kmeans.cluster_centers_
        som.set_weights_with_data(centroids)"""
        # Assuming sample_encoded_data is an array of encoded data points
        sample_encoded_data = np.array(sample_encoded_data)

        # Let's say your SOM grid is m x n

        # Calculate the total number of weights needed
        # Calculate the total number of weights needed
        total_weights = self.m * self.n   
        
        pca = PCA(n_components=2)  # Adjust n_components based on your SOM size and data
        principal_components = pca.fit_transform(sample_encoded_data)

        # Let's say your SOM grid is m x n and you're using the first two PCs
        weights = np.zeros((self.m * self.n, sample_encoded_data.shape[1]))  # Initialize SOM weights
        # Linearly distribute the first two PCs across the SOM grid
        for i in range(m):
            for j in range(n):
                # Calculate coefficients for linear combinations
                coeff1 = i / (m - 1)
                coeff2 = j / (n - 1)
                # Combine the PCs to initialize the neuron
                neuron_position = coeff1 * pca.components_[0] + coeff2 * pca.components_[1]
                weights[i * n + j] = neuron_position
        som.set_weights_with_data(weights)

        """# Randomly select 'total_weights' points from the encoded data
        random_indices = np.random.choice(sample_encoded_data.shape[0], size=total_weights, replace=False)
        selected_data_points = sample_encoded_data[random_indices]

        # Use these selected data points as the initial weights for the SOM
        som.set_weights_with_data(selected_data_points)"""

        initialized_som = True
        print('SOM INITIALIZED')
        # Custom training loop
        for epoch in range(self.ae_epochs):
            start_time = time.time()

            #prepare a list to accumulate encoded data for SOM training
            all_encoded_data = []
            
            T = self.decay_fcn(self.ae_epochs, epoch, self.T)
            for step, batch_data in enumerate(dataset):
                with tf.GradientTape(persistent=True) as tape:
                    # Forward pass through the autoencoder
                    encoded = encoder(batch_data, training=True)
                    decoded = decoder(encoded, training=True)

                    """if not initialized_som:
                        sample_encoded_data.extend(encoded.numpy().reshape(-1, self.encoding_dim))
                        if len(sample_encoded_data) >= 20000:  # E.g., desired_sample_size = 10000
                            # Perform K-Means clustering
                            kmeans = KMeans(n_clusters=self.m * self.n, random_state=0).fit(sample_encoded_data)
                            centroids = kmeans.cluster_centers_
                            som.set_weights_with_data(centroids)
                            initialized_som = True
                            print('SOM INITIALIZED')"""

                    # Reconstruction loss
                    reconstruction_loss = tf.reduce_mean(tf.square(tf.cast(batch_data, tf.float32) - tf.cast(decoded, tf.float32)))


                    som_true_loss = som._loss(encoded, T)

                    som_loss = self.alpha_loss * som_true_loss
                    
                    encoder_loss = reconstruction_loss + som_loss

                """# Calculate validation loss
                val_loss = reconstruction_loss.numpy()

                # Check if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_count = 0
                else:
                    patience_count += 1

                # Stop training if patience is exceeded
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch+1}, validation loss did not improve for {patience} epochs.")
                    break"""
                alpha = self.alpha_loss 
                # Compute and apply gradients for the autoencoder
                som_gradients = tape.gradient(som_loss, som.trainable_variables)
                encoder_gradients = tape.gradient(encoder_loss, encoder.trainable_variables)
                decoder_gradients = tape.gradient(reconstruction_loss, decoder.trainable_variables)
                
                grads_and_vars = list(chain(zip(decoder_gradients, decoder.trainable_variables),
                            zip(encoder_gradients, encoder.trainable_variables),
                            zip(som_gradients, som.trainable_variables)))


                
                opt.apply_gradients(grads_and_vars)
                #opt.apply_gradients(zip(encoder_gradients, encoder.trainable_variables))
                #opt.apply_gradients(zip(som_gradients, som.trainable_variables))

                

            print(f'\nEPOCH:{epoch}\nTOTAL LOSS: {encoder_loss}\n SOM LOSS: {som_loss}\nSOM TRUE LOSS: {som_true_loss}\n RECONSTRUCTION LOSS: {reconstruction_loss}\n TRUE SOM LOSS TO T RATIO {som_true_loss/T}')
            # Print loss information or perform evaluations at the end of each epoch
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"EPOCH RUNTIME: {execution_time} s")
            if epoch % 50 == 0:
                model_save_path = os.path.join(output_dir, 'decoder_model.h5')
                som_save_path = os.path.join(output_dir, 'som_model')
                encoder_save_path = os.path.join(output_dir, 'encoder_model.h5')  # Path to save the encoder
                decoder.save(model_save_path)
                encoder.save(encoder_save_path)  # Saving the encoder
                encoded = encoder.predict(self.data)
                som_placement = som.find_batch_bmus(encoded)
                save_som_model(som, som_save_path)
                print(f"Autoencoder Model saved to {model_save_path}")
                print(f"Encoder Model saved to {encoder_save_path}")
                args = (som_placement, self.data_matrix, epoch)
                plot_checkpoint(som_placement, self.data_matrix, epoch)
            sys.stdout.flush()


        model_save_path = os.path.join(output_dir, 'autoencoder_model.h5')
        som_save_path = os.path.join(output_dir, 'som_model')
        encoder_save_path = os.path.join(output_dir, 'encoder_model.h5')  # Path to save the encoder
        autoencoder.save(model_save_path)
        encoder.save(encoder_save_path)  # Saving the encoder
        save_som_model(som, som_save_path)
        print(f"Autoencoder Model saved to {model_save_path}")
        print(f"Encoder Model saved to {encoder_save_path}")
    
        return autoencoder, encoder  # Returning both models




class SOM(layers.Layer):
    def __init__(self, m, n, dim, alpha, sigma, batch_size, **kwargs):
        super(SOM, self).__init__(**kwargs)
        self.m = m
        self.n = n
        self.dim = dim
        self.alpha = alpha
        self.sigma = sigma
        self.batch_size = batch_size
        self.locations = tf.Variable(self._generate_locations(m, n), dtype=tf.float32, trainable = False)
        self.som_weights = self.add_weight(shape=(m * n, dim), initializer='random_normal', trainable=True)

    def _generate_locations(self, m, n):
        return np.array([np.array([i, j]) for i in range(m) for j in range(n)])

    def call(self, inputs):
        bmu_indices = self.find_bmus(inputs)
        self.add_loss(self.som_loss(inputs))
        return self.som_weights

    def find_bmus(self, inputs):
        # Compute Euclidean distances from all inputs to all weight vectors
        expanded_inputs = tf.cast(tf.expand_dims(inputs, axis=1), tf.float32)  # Shape: [batch_size, 1, dim]
        expanded_weights = tf.cast(tf.expand_dims(self.som_weights, axis=0), tf.float32)  # Shape: [1, m*n, dim]
        distances = tf.reduce_sum(tf.square(expanded_inputs - expanded_weights), axis=2)  # Shape: [batch_size, m*n]
        bmu_indices = tf.argmin(distances, axis=1)  # Shape: [batch_size]
        return bmu_indices
    
    def find_batch_bmus(self, inputs, batch_size=256):
        num_batches = np.ceil(inputs.shape[0] / batch_size).astype(int)
        bmu_indices = []

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = (batch_idx + 1) * batch_size
            batch_inputs = inputs[batch_start:batch_end]

            batch_bmu_indices = self.find_bmus(batch_inputs)
            bmu_indices.append(batch_bmu_indices)

        bmu_indices = tf.concat(bmu_indices, axis=0)
        return bmu_indices

    def set_weights_with_data(self, data_points):
        """
        Initialize the SOM weights with given data points.
        """
        if data_points.shape[0] != self.m * self.n:
            raise ValueError("The number of data points must match m*n.")
        new_weights = tf.cast(data_points, dtype=tf.float32)  # Cast to float32 to match som_weights dtype
        new_weights = tf.reshape(new_weights, (self.m * self.n, self.dim))  # Use tf.reshape to adjust the tensor shape
        self.som_weights.assign(new_weights)

    
    def _loss(self, inputs, T):
        bmu_indices = self.find_bmus(inputs)
        # Get BMU locations in the grid
        bmu_locations = tf.gather(self.locations, bmu_indices)  # Shape: [batch_size, 2]
        
        # Expand SOM grid locations to compare against each BMU location
        expanded_locations = tf.reshape(self.locations, [self.m, self.n, 1, 2])  # Shape: [m, n, 1, 2]
        
        # Compute Manhattan distances
        manhattan_distances = tf.reduce_sum(tf.abs(expanded_locations - bmu_locations), axis=3)  # Shape: [m, n, batch_size]
        manhattan_distances = tf.transpose(manhattan_distances, perm=[2, 0, 1])  # Transpose to shape: [batch_size, m, n]
        influence = tf.exp(-manhattan_distances**2 / T**2)
        # Now, let's work on getting delta_input_prototypes in the desired shape [batch_size, m, n]
        # First, calculate the difference between inputs and weights without reducing sum
        
        # Now, you want to sum across the dim dimension to collapse it, but keep the batch_size and m*n structure
        # Reshape weights to have a grid structure for broadcasting
        reshaped_weights = tf.reshape(self.som_weights, [self.m, self.n, self.dim])
        
        # Expand dimensions for inputs and weights to enable broadcasting
        expanded_inputs = tf.cast(tf.expand_dims(tf.expand_dims(inputs, 1), 2), tf.float32)  # New shape: [batch_size, 1, 1, dim]
        expanded_weights = tf.cast(tf.expand_dims(reshaped_weights, 0), tf.float32)  # New shape: [1, m, n, dim]
        
        # Compute absolute differences and then sum over the feature dimension to collapse it
        absolute_differences = tf.square(expanded_inputs - expanded_weights)
        summarized_differences = tf.reduce_sum(absolute_differences, axis=-1)            
        
        loss_term = summarized_differences * influence
        loss_sum1 = tf.reduce_sum(loss_term, [0, 1])
        loss = tf.reduce_mean(loss_sum1)
        return loss
    
    def plot_umatrix(self,  color_map='gray'):
        size=(self.m, self.n)
        """
        Plots the U-matrix of the SOM.
        :param size: Tuple of the figure size.
        :param color_map: Color map to use for plotting.
        """
        # Reshape SOM weights to a grid
        weight_grid = tf.reshape(self.som_weights, (self.m, self.n, self.dim))
        
        # Initialize U-matrix
        umatrix = np.zeros((self.m, self.n))
        
        for i in range(self.m):
            for j in range(self.n):
                dist = 0
                count = 0
                
                # Calculate the distance to the neighboring neurons (Manhattan distance)
                if i > 0:  # Up
                    dist += tf.norm(weight_grid[i, j, :] - weight_grid[i-1, j, :])
                    count += 1
                if i < self.m-1:  # Down
                    dist += tf.norm(weight_grid[i, j, :] - weight_grid[i+1, j, :])
                    count += 1
                if j > 0:  # Left
                    dist += tf.norm(weight_grid[i, j, :] - weight_grid[i, j-1, :])
                    count += 1
                if j < self.n-1:  # Right
                    dist += tf.norm(weight_grid[i, j, :] - weight_grid[i, j+1, :])
                    count += 1
                
                # Average distance
                umatrix[i, j] = dist / count
        
        plt.figure(figsize=size)
        plt.imshow(umatrix, cmap=color_map, interpolation='nearest')
        plt.colorbar()
        plt.title('U-Matrix')
        plt.savefig('umatrix.png')


def save_som_model(som_layer, save_path):
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save the weights
    weights_path = os.path.join(save_path, 'som_weights.npy')
    np.save(weights_path, som_layer.som_weights.numpy())
    
    # Save other necessary configurations if there are any
    config_path = os.path.join(save_path, 'som_config.json')
    with open(config_path, 'w') as f:
        config = {
            "m": som_layer.m,
            "n": som_layer.n,
            "dim": som_layer.dim,
            # Add other configurations as necessary
        }
        json.dump(config, f)
    
    print(f"SOM model saved at {save_path}")




print('Fcns and Class Initialization Complete')

print('EXTRACTING DATA')
try:
    multiprocessing.set_start_method('fork')  # or 'forkserver'
except:
    True == True
extractor = FITSDataExtractor('highz_test/')
data_matrix = extractor.parallel_open_files()
data = np.array([row for row in data_matrix['interpol_flux']], dtype=np.float32)  # Cast to float32

with h5py.File('all_flux.h5', 'w') as f:
    # Create a dataset in the file
    f.create_dataset('flux', data=data)

print(f'NUMBER OF TRAINING POINTS: {len(data)}')
"""with h5py.File('/users/coldatom/code/pca/all_flux.h5', 'r') as f:
    # Assuming 'array_data' is the name of your dataset in the HDF5 file
    data = np.array(f['flux'])
data_matrix = True"""

ae_epochs = 3000
som_epochs = 1
batch_size = 256
ae_lr = 0.00005
som_lr = 0.00005
alpha = 0.00005
T = 100
sigma = 100
som_batches = batch_size
alpha_loss = 5e-3
T_min = 0.1
encoding_dim = 256
ae = autoencoder(data, data_matrix, encoding_dim = encoding_dim, som_width=64, som_height= 64,T_min = T_min, ae_epochs = ae_epochs, som_epochs = som_epochs, batch_size = batch_size, ae_lr = ae_lr, som_lr = som_lr, alpha = alpha, T = T, sigma = sigma, som_batches = som_batches)


model, encoder = ae.train_and_evaluate_autoencoder('ai_test/highz_autoencoder/')

print('training complete')