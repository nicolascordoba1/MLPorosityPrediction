import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_results(impedance, y_predict, y, slice, prediction=True):   

    #Plots inician acá
    if prediction is False:
        fig, ax = plt.subplots(1, 2, figsize=(20,10))
    else:
        fig, ax = plt.subplots(1, 4, figsize=(20,10))
        error = np.mean(np.square(y_predict-y)**2)
        print(f'El error es {error}')
        if slice==False:
            slice = np.random.randint(0,y_predict.shape[0])
        print(f'Las dimensiones del Y predict es {y_predict.shape}')
        print(f'Las dimensiones del Y normal es {y.shape}')
        
        error_slice = y_predict[slice,:,:] - y[slice,:,:] 
    #Impedancia
    im = ax[0].imshow(impedance[slice,:,:])
    ax[0].set_title(f'Impedance. slice: {slice}')
    fig.colorbar(im, ax=ax[0], shrink=0.5)
    
    im2 = ax[1].imshow(y[slice,:,:], vmin=0, vmax=0.2)
    ax[1].set_title(f'Porosity Original. slice: {slice}')
    fig.colorbar(im2, ax=ax[1], shrink=0.5)
    
    if prediction is False:
        fig.tight_layout()
        plt.show()
    else:
        im3 = ax[2].imshow(y_predict[slice,:,:], vmin=0, vmax=0.2)
        ax[2].set_title(f'Porosity Prediction. slice: {slice}')
        fig.colorbar(im2, ax=ax[2], shrink=0.5)
        
        #ColorMap
        custom_map = LinearSegmentedColormap.from_list('custom', ['red', 'orange','yellow', 'black', 'yellow', 'orange', 'red'])
        #Plot Error
        im4 = ax[3].imshow(error_slice, cmap=custom_map, vmin=-0.1, vmax=0.1)
        ax[3].set_title(f'Difference Porosity. slice: {slice}')
        fig.colorbar(im4, ax=ax[3], shrink=0.5)
        
        fig.tight_layout()
        
        #fig.savefig(f'/kaggle/working/porosity_prediction_slice_{slice}.png')
        
        plt.show()