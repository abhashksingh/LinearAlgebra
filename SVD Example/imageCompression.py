from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16,8]

image_path = 'abhash.jpg'
if os.path.exists(image_path):
    A = imread(image_path)
    print("Image loaded successfully")
    X = np.mean(A,-1) #convert RGB to greyscale
    img = plt.imshow(X)
    img.set_cmap('gray')
    plt.axis('off')
    plt.savefig("abhash_b&w")
    plt.show()

    U,S,VT = np.linalg.svd(X,full_matrices=False)

    S = np.diag(S)

    j = 0
    for r in (5,20,100):
        Xapprox = U[:,:r] @S[0:r,:r] @VT[:r,:]
        plt.figure(j+1)
        j+=1
        img = plt.imshow(Xapprox)
        img.set_cmap('gray')
        plt.axis('off')
        plt.show()
    
    plt.figure(1)
    plt.semilogy(np.diag(S))
    plt.title('Singular Values')
    plt.show()

    plt.figure(2)
    plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
    plt.title('Singular Values Cumulative Sum')
    plt.show()
    
else:
    print(f"File not found: {image_path}")
    print("Current directory:", os.getcwd())
    print("Files in current directory:", os.listdir())


