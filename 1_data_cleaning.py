import re
import pandas as pd
import numpy as np
import random

import segyio

from skimage.transform import resize

import sporco

porosity_inline_boreas1 = 'data_poseidon/lineas/boreas_1_inline_porosity.segy'
porosity_xline_boreas1 = 'data_poseidon/lineas/boreas_1_xline_porosity.segy'
porosity_xline_poseidon1 = 'data_poseidon/lineas/poseidon_1_xline_porosity.segy'
porosity_inline_poseidon1 = 'data_poseidon/lineas/poseidon_1_inline_porosity.segy'
porosity_xline_poseidon2 = 'data_poseidon/lineas/poseidon_2_xline_porosity.segy'
porosity_inline_poseidon2 = 'data_poseidon/lineas/poseidon_2_inline_porosity.segy'

inversion_inline_boreas1 = 'data_poseidon/lineas/boreas_1_inline_inversion.segy'
inversion_xline_boreas1 = 'data_poseidon/lineas/boreas_1_xline_inversion.segy'
inversion_xline_poseidon1 = 'data_poseidon/lineas/poseidon_1_xline_inversion.segy'
inversion_inline_poseidon1 = 'data_poseidon/lineas/poseidon_1_inline_inversion.segy'
inversion_xline_poseidon2 = 'data_poseidon/lineas/poseidon_2_xline_inversion.segy'
inversion_inline_poseidon2 = 'data_poseidon/lineas/poseidon_2_inline_inversion.segy'

def parse_trace_headers(segyfile, n_traces):
    '''
    Parse the segy file trace headers into a pandas dataframe.
    Column names are defined from segyio internal tracefield
    One row per trace
    '''
    # Get all header keys
    headers = segyio.tracefield.keys
    # Initialize dataframe with trace id as index and headers as columns
    df = pd.DataFrame(index=range(1, n_traces + 1),
                      columns=headers.keys())
    # Fill dataframe with all header values
    for k, v in headers.items():
        df[k] = segyfile.attributes(v)[:]
    return df

def parse_text_header(segyfile):
    '''
    Format segy text header into a readable, clean dict
    '''
    raw_header = segyio.tools.wrap(segyfile.text[0])
    # Cut on C*int pattern
    cut_header = re.split(r'C ', raw_header)[1::]
    # Remove end of line return
    text_header = [x.replace('\n', ' ') for x in cut_header]
    text_header[-1] = text_header[-1][:-2]
    # Format in dict
    clean_header = {}
    i = 1
    for item in text_header:
        key = "C" + str(i).rjust(2, '0')
        i += 1
        clean_header[key] = item
    return clean_header

lines_porosity = [porosity_inline_boreas1,
                  porosity_inline_poseidon1,
                  porosity_inline_poseidon2,
                  porosity_xline_boreas1,
                  porosity_xline_poseidon1,                  
                  porosity_xline_poseidon2,
                  ]

lines_impedance = [inversion_inline_boreas1,
                   inversion_inline_poseidon1,
                   inversion_inline_poseidon2,
                   inversion_xline_boreas1,
                   inversion_xline_poseidon1,
                   inversion_xline_poseidon2
                   ]

lines_array_porosity = np.empty(shape=(491,452,6))
lines_array_impedance = np.empty(shape=(491,452,6))

for index, line in enumerate(lines_porosity):
    with segyio.open(line, ignore_geometry=True) as f:
        data = f.trace.raw[:]  # Get all data into memory (could cause on big files)
        resized_image = resize(data, (491, 452), anti_aliasing=True)
        lines_array_porosity[:,:,index] = resized_image

        del data
        del resized_image
        
for index, line in enumerate(lines_impedance):
    with segyio.open(line, ignore_geometry=True) as f:
        data = f.trace.raw[:]  # Get all data into memory (could cause on big files)
        resized_image = resize(data, (491, 452), anti_aliasing=True)
        lines_array_impedance[:,:,index] = resized_image
        
        del data
        del resized_image
    
lines_array_impedance = lines_array_impedance[:,:250,:]
lines_array_porosity = lines_array_porosity[:,:250,:]

print(f"Las dimensiones de la impedancia son: {lines_array_impedance.shape}")
print(f"Las dimensiones de la porosidad son: {lines_array_porosity.shape}")



def min_max_scale(x, min, max):

  x_std = (x - min) / (max - min)
  x_scaled = x_std * 2 - 1
  return x_scaled

def inverse_min_max_scale(x, min, max):

  x_normalized = (x + 1) / 2
  x_unscaled = x_normalized * (max - min) + min
  return x_unscaled


impedance_max=np.max(lines_array_impedance) 
impedance_min=np.min(lines_array_impedance) 

print(impedance_max, impedance_min)

#impedance_normalized = (lines_array_impedance - np.min(lines_array_impedance))/(np.max(lines_array_impedance)-np.min(lines_array_impedance))

print('Entré a blocko')

data_blocks_porosity = sporco.array.extract_blocks(lines_array_porosity, (32,32), (16,))
data_blocks_impedance = sporco.array.extract_blocks(lines_array_impedance, (32,32), (16,))

print('Salí de blocko')

print(data_blocks_porosity.shape, data_blocks_impedance.shape)

""" Este bloque de código asegura que las dimensiones de los arrays sean compatibles con como entran 
los datos a la red neuronal. Se organiza de tal forma que la primera dimension es la cantidad de 
imagenes que hay, la segunda es el numero de canales y la tercera es el ancho y alto de la figura"""

new_impedance = np.moveaxis(data_blocks_impedance, -1, 0)
new_new_impedance = np.moveaxis(new_impedance, 1, 2)

new_porosity = np.moveaxis(data_blocks_porosity, -1, 0)
new_new_porosity = np.moveaxis(new_porosity, 1, 2)

print(new_new_impedance.shape)

print(new_impedance.shape)

numbers = list(range(new_new_impedance.shape[0]))

# Shuffle the list using random.shuffle
random_index = random.shuffle(numbers)

# Convert the shuffled list to a NumPy array
random_index = np.array(numbers)

new_new_impedance = new_new_impedance[random_index]
new_new_porosity = new_new_porosity[random_index]

np.save('data_poseidon/processed/impedance_blocked.npy', new_new_impedance)
np.save('data_poseidon/processed/porosity_blocked.npy', new_new_porosity)


