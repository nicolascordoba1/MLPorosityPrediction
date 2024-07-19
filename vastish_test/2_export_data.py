import numpy as np
import segyio

# File paths
original_segy_file = "./data_decatur/seismic/porosity.segy"
new_segy_file = "./data_decatur/seismic/porosity_modeled.segy"

#Import Data
phi_pred_exploracion = np.load('./data_decatur/processed/porosity_modeled.npy')

# Convert your seismic data to float32
segyout = phi_pred_exploracion.astype(np.float32)

# Original subset dimensions
ilines, xlines, n_samples = segyout.shape
n_traces = ilines * xlines
segyout = segyout.reshape(n_traces, n_samples)
dtout = 20

# Load the original SEGY file

with segyio.open(original_segy_file, "r") as segyfile:
    print('Is inline the fast mode? ' + str(segyfile.fast is segyfile.iline))
  
    # Create the specification for the new SEGY file
    spec = segyio.spec()
    spec.sorting = segyfile.sorting
    spec.format = segyfile.format
    spec.samples = list(range(4530, 4530 + n_samples * dtout,  dtout))  # Depth range
    spec.ilines = range(ilines)  # Define new ilines range
    spec.xlines = range(930, 930 + xlines)  # Define new xlines range based on slicing
    spec.tracecount = n_traces
    
    print(spec.ilines)

    # Write the new SEGY file
    with segyio.create(new_segy_file, spec) as f:
        f.text[0] = segyfile.text[0]
        f.bin = segyfile.bin
        f.bin[segyio.BinField.Interval] = dtout  # Set the sample interval (depth interval)

        # Set coordinate scale factor to 100 (for 0.01 scaling)
        #f.bin[segyio.BinField.SourceGroupScalar] = 100  

        # Loop through traces and set headers
        for i in range(n_traces):
            # Assign trace data
            f.trace[i] = segyout[i]

            # Calculate inline and crossline indices for the subset
            inline_idx = i // xlines
            crossline_idx = i % xlines

            # Map to the correct inline and crossline numbers from the original SEGY file
            inline = segyfile.ilines[inline_idx]
            crossline = 930 + crossline_idx
            
            # Calculate header index for the original file
            original_trace_index = (inline - segyfile.ilines[0]) * len(segyfile.xlines) + (crossline - segyfile.xlines[0])

            # Retrieve original CDP_X and CDP_Y, divide by 100, and convert to integer
            original_cdp_x = int(segyfile.header[original_trace_index][segyio.TraceField.CDP_X]/100) 
            original_cdp_y = int(segyfile.header[original_trace_index][segyio.TraceField.CDP_Y]/100)

            # Set headers using the calculated indices and original file's header values
            f.header[i] = {
                segyio.TraceField.INLINE_3D: inline,
                segyio.TraceField.CROSSLINE_3D: crossline,
                segyio.TraceField.CDP_X: original_cdp_x,
                segyio.TraceField.CDP_Y: original_cdp_y,
            }

print("SEGY file created successfully.")