from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16,8]

image_path = 'abhash.jpg'
if os.path.exists(image_path):
    # Load the color image
    A = imread(image_path)
    print("Image loaded successfully")
    
    # Determine if image is in 0-1 range or 0-255 range
    is_float_image = A.dtype == np.float32 or A.dtype == np.float64
    
    # Store original data type for later
    original_dtype = A.dtype
    
    # Ensure image is float in 0-1 range for processing
    if not is_float_image:
        A_normalized = A.astype(float) / 255.0
    else:
        A_normalized = A.copy()
    
    # Display original image
    plt.imshow(A)
    plt.axis('off')
    plt.title('Original Color Image')
    plt.savefig("abhash_original")
    plt.show()

    # Initialize arrays to store the SVD results for each channel
    height, width, channels = A.shape
    
    # Create empty arrays to store the results
    U_channels = []
    S_channels = []
    VT_channels = []
    
    # Perform SVD on each color channel
    for i in range(channels):
        U, S, VT = np.linalg.svd(A_normalized[:,:,i], full_matrices=False)
        U_channels.append(U)
        S_channels.append(S)
        VT_channels.append(VT)
    
    # Function to reconstruct image with r singular values
    def reconstruct_image(r):
        reconstructed = np.zeros_like(A_normalized)
        for i in range(channels):
            S_diag = np.diag(S_channels[i])
            reconstructed[:,:,i] = U_channels[i][:,:r] @ S_diag[0:r,:r] @ VT_channels[i][:r,:]
        
        # Clip values to ensure they stay in valid range
        reconstructed = np.clip(reconstructed, 0, 1)
        
        # Convert back to original data type if needed
        if not is_float_image:
            reconstructed = (reconstructed * 255).astype(original_dtype)
            
        return reconstructed
    
    # Display reconstructions at different ranks
    for j, r in enumerate([5, 20, 100]):
        Xapprox = reconstruct_image(r)
        plt.figure(j+1)
        plt.imshow(Xapprox)
        plt.axis('off')
        plt.title(f'Color Image with r={r} singular values')
        plt.savefig(f"abhash_color_r{r}")
        plt.show()
    
    # Plot singular values for each channel
    plt.figure()
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.semilogy(S_channels[i], label=color, color=color)
    plt.title('Singular Values for RGB Channels')
    plt.legend()
    plt.savefig("abhash_singular_values")
    plt.show()

    # Plot cumulative energy for each channel
    plt.figure()
    for i, color in enumerate(['red', 'green', 'blue']):
        plt.plot(np.cumsum(S_channels[i])/np.sum(S_channels[i]), 
                 label=color, color=color)
    plt.title('Singular Values Cumulative Sum for RGB Channels')
    plt.legend()
    plt.xlabel('Number of Singular Values')
    plt.ylabel('Cumulative Energy')
    plt.grid(True)
    plt.savefig("abhash_cumulative_energy")
    plt.show()
    
    # Calculate compression ratio
    original_size = A.size
    for r in [5, 20, 100]:
        compressed_size = r * (height + width + 1) * channels
        ratio = original_size / compressed_size
        print(f"Compression ratio with r={r}: {ratio:.2f}x")
    
else:
    print(f"File not found: {image_path}")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir())