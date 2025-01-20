import numpy as np
import segyio

# File paths
original_segy_file = "data/data_decatur/seismic/seismic_model.segy"
new_segy_file = "data/data_decatur/seismic/porosity_modeled_explo_new.segy"

# Import Data
phi_pred_exploracion = np.load('data/data_decatur/processed/porosity_modeled_exploration_block.npy')

# Convert your seismic data to float32
segyout = phi_pred_exploracion.astype(np.float32)

# Original subset dimensions
ilines, xlines, n_samples = segyout.shape
n_traces = ilines * xlines
segyout = segyout.reshape(n_traces, n_samples)
dtout = 20  # Sampling interval in feet

# Define depth range
start_depth = 4700  # Starting depth in feet
manual_depths = list(range(start_depth, start_depth + n_samples * dtout, dtout))  # Depth range

# Load the original SEGY file
with segyio.open(original_segy_file, "r") as segyfile:
    print('Is inline the fast mode? ' + str(segyfile.fast is segyfile.iline))

    # Create the specification for the new SEGY file
    spec = segyio.spec()
    spec.sorting = segyfile.sorting
    spec.format = segyfile.format
    spec.samples = manual_depths  # Set depth range explicitly
    spec.ilines = range(ilines)  # Define new ilines range
    spec.xlines = range(930, 930 + xlines)  # Define new xlines range based on slicing
    spec.tracecount = n_traces

    print("Spec samples (first 10):", spec.samples[:10])  # Debugging
    print("Number of samples:", len(spec.samples))

    # Write the new SEGY file
    with segyio.create(new_segy_file, spec) as f:
        f.text[0] = segyfile.text[0]
        f.bin = segyfile.bin
        f.bin[segyio.BinField.Interval] = dtout  # Set the sample interval (depth interval)

        # Explicitly set trace header values for each trace
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

            # Retrieve original CDP_X and CDP_Y
            original_cdp_x = int(segyfile.header[original_trace_index][segyio.TraceField.CDP_X] / 100)
            original_cdp_y = int(segyfile.header[original_trace_index][segyio.TraceField.CDP_Y] / 100)

            # Set headers using the calculated indices and original file's header values
            f.header[i] = {
                segyio.TraceField.INLINE_3D: inline,
                segyio.TraceField.CROSSLINE_3D: crossline,
                segyio.TraceField.CDP_X: original_cdp_x,
                segyio.TraceField.CDP_Y: original_cdp_y,
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,  # Number of samples per trace
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: dtout * 1000,  # Convert to microseconds for SEG-Y standard
            }

print("SEGY file created successfully.")

# Debugging the newly created SEGY file
with segyio.open(new_segy_file, "r") as porosity_segy:

    print("Number of traces:", porosity_segy.tracecount)
    print("Length of each trace:", porosity_segy.samples.size)
    print("Range of depth(ft):", porosity_segy.samples[0], "-", porosity_segy.samples[-1])

    inlines = porosity_segy.attributes(segyio.TraceField.INLINE_3D)[:]
    print("Inline min:", inlines.min())  
    print("Inline max:", inlines.max()) 
    xlines = porosity_segy.attributes(segyio.TraceField.CROSSLINE_3D)[:]
    print("Crossline min:", xlines.min())  
    print("Crossline max:", xlines.max()) 
