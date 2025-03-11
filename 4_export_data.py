import numpy as np
import segyio
import matplotlib.pyplot as plt
import os

def convert_npy_to_segy(
    npy_file_path,
    output_segy_path,
    template_segy_path=None,
    start_depth=4700,
    sample_interval=20,
    il_range=None,
    xl_range=None
):
    """
    Convert a 3D numpy array to SEGY format.
    
    Parameters:
    -----------
    npy_file_path : str
        Path to the .npy file containing 3D seismic data (ilines, xlines, samples)
    output_segy_path : str
        Path where the new SEGY file will be written
    template_segy_path : str, optional
        Path to a template SEGY file to copy header information from
    start_depth : int, default=4700
        Starting depth in feet
    sample_interval : int, default=20
        Sampling interval in feet
    il_range : tuple, optional
        Range of inline numbers (start, end). If None, will use sequential numbering.
    xl_range : tuple, optional
        Range of crossline numbers (start, end). If None, will use sequential numbering.
    """
    # Load numpy data
    print(f"Loading data from {npy_file_path}")
    data_3d = np.load(npy_file_path)
    
    # Check data values and range before writing
    print("Data shape:", data_3d.shape)
    print("Data min/max:", np.min(data_3d), np.max(data_3d))
    print("Data mean/std:", np.mean(data_3d), np.std(data_3d))
    
    # Convert data to float32 (required for SEG-Y)
    segy_data = data_3d.astype(np.float32)
    
    # Extract dimensions
    n_ilines, n_xlines, n_samples = segy_data.shape
    n_traces = n_ilines * n_xlines
    
    # Reshape for SEGY format (traces x samples)
    segy_data = segy_data.reshape(n_traces, n_samples)
    
    # Define depth range
    depth_range = np.arange(start_depth, start_depth + n_samples * sample_interval, sample_interval)
    
    # Define inline and crossline numbering
    # Start inline from 53 and increment by 1
    ilines = np.arange(53, 53 + n_ilines, dtype=int)
    
    # Start crossline from 930 and increment by 1
    xlines = np.arange(930, 930 + n_xlines, dtype=int)
    
    # Initialize spec
    spec = segyio.spec()
    spec.samples = depth_range
    spec.ilines = ilines
    spec.xlines = xlines
    spec.format = 5  # IEEE float32
    spec.sorting = 1  # INLINE_SORTING
    spec.tracecount = n_traces
    
    # Copy binary header from template if provided
    binary_header = {}
    if template_segy_path and os.path.exists(template_segy_path):
        print(f"Using template SEGY file: {template_segy_path}")
        with segyio.open(template_segy_path, "r") as template:
            # Copy binary header
            binary_header = dict(template.bin)
            
            # Copy text header if needed
            text_header = template.text[0]
            
            # Get CDP coordinates from template if possible
            try:
                cdp_x = template.attributes(segyio.TraceField.CDP_X)[:]
                cdp_y = template.attributes(segyio.TraceField.CDP_Y)[:]
                has_coords = True
                # Check if we have enough coordinates
                if len(cdp_x) < n_traces:
                    print(f"Warning: Template has {len(cdp_x)} traces but output needs {n_traces}")
                    has_coords = False
            except Exception as e:
                print(f"Could not extract coordinates from template: {e}")
                has_coords = False
                
            # Check sorting in template
            print('Template sorting:', template.sorting)
            spec.sorting = template.sorting
    else:
        # Create a default text header
        text_header = ' ' * 3200  # 3200 bytes of space
        text_header = text_header.encode('ascii')
        has_coords = False
        
    # Update binary header with key values
    binary_header[segyio.BinField.Interval] = sample_interval * 1000  # Convert to microseconds
    binary_header[segyio.BinField.Samples] = n_samples
    binary_header[segyio.BinField.MeasurementSystem] = 2  # 1=meters, 2=feet
    binary_header[segyio.BinField.Format] = 5  # IEEE float
    
    # Create the output SEGY file
    print(f"Creating SEGY file: {output_segy_path}")
    with segyio.create(output_segy_path, spec) as f:
        # Write text header
        f.text[0] = text_header
        
        # Write binary header
        f.bin = binary_header
        
        # Write trace data and headers
        for i in range(n_traces):
            # Write trace data
            f.trace[i] = segy_data[i]
            
            # Calculate inline and crossline indices
            il_idx = i // n_xlines
            xl_idx = i % n_xlines
            
            # Create trace header
            header = {
                segyio.TraceField.INLINE_3D: ilines[il_idx],
                segyio.TraceField.CROSSLINE_3D: xlines[xl_idx],
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: sample_interval * 1000,  # In microseconds
                segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                segyio.TraceField.ElevationScalar: -100,
                segyio.TraceField.SourceGroupScalar: -100,
            }
            
            # Add coordinates if available from template
            if has_coords and i < len(cdp_x):
                # Apply scaling directly to coordinates (divide by 100)
                scaled_x = cdp_x[i]
                scaled_y = cdp_y[i]
                header[segyio.TraceField.CDP_X] = scaled_x
                header[segyio.TraceField.CDP_Y] = scaled_y
            else:
                # Generate placeholder coordinates 
                # These are arbitrary coordinates that increase with inline/crossline
                header[segyio.TraceField.CDP_X] = ilines[il_idx] * 100
                header[segyio.TraceField.CDP_Y] = xlines[xl_idx] * 100
            
            # Set coordinate type to rectangular (byte 89)
            try:
                # Access the binary trace header directly
                if hasattr(segyio.TraceField, 'CoordinateUnits'):
                    header[segyio.TraceField.CoordinateUnits] = 1  # 1=Local grid
            except Exception as e:
                print(f"Could not set coordinate units: {e}")
                
            # SEG-Y Rev 1 specific - try to handle byte 89
            try:
                # Some versions allow direct key access
                header[89] = 1  # Set byte 89 to 1 (rectangular)
            except Exception:
                pass
                
            # Write header
            f.header[i] = header
    
    print("SEGY file created successfully.")
    
    # Validate the newly created SEGY file
    validate_segy(output_segy_path)
    
    # Visualize data
    visualize_segy(output_segy_path)
    
def validate_segy(segy_path):
    """
    Validate a SEGY file and print information about its contents.
    """
    print(f"\nValidating SEGY file: {segy_path}")
    with segyio.open(segy_path, "r") as segy:
        # Access headers and dimensions
        print("Number of traces:", segy.tracecount)
        print("Samples per trace:", segy.samples.size)
        print("Depth range (ft):", segy.samples[0], "-", segy.samples[-1])
        
        # Check inlines and crosslines
        inlines = segy.attributes(segyio.TraceField.INLINE_3D)[:]
        unique_inlines = np.unique(inlines)
        print("Inline range:", inlines.min(), "-", inlines.max())
        print("Number of unique inlines:", len(unique_inlines))
        
        xlines = segy.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        unique_xlines = np.unique(xlines)
        print("Crossline range:", xlines.min(), "-", xlines.max())
        print("Number of unique crosslines:", len(unique_xlines))
        
        # Check data values in the output file
        trace_data = segy.trace.raw[:]
        print("Output data min/max:", np.min(trace_data), np.max(trace_data))
        
        # Check coordinates
        try:
            cdp_x = segy.attributes(segyio.TraceField.CDP_X)[:]
            cdp_y = segy.attributes(segyio.TraceField.CDP_Y)[:]
            print("CDP_X range:", np.min(cdp_x), "-", np.max(cdp_x))
            print("CDP_Y range:", np.min(cdp_y), "-", np.max(cdp_y))
        except Exception as e:
            print(f"Could not read coordinates: {e}")
    
def visualize_segy(segy_path, save_path=None):
    """
    Visualize a slice from the SEGY file.
    """
    print(f"\nVisualizing SEGY file: {segy_path}")
    if save_path is None:
        save_path = os.path.splitext(segy_path)[0] + "_validation.png"
        
    plt.figure(figsize=(12, 10))
    with segyio.open(segy_path, "r") as segy:
        # Get a center inline slice
        cube = segyio.tools.cube(segy)
        
        # Determine which dimension to slice
        inline_count = len(np.unique(segy.attributes(segyio.TraceField.INLINE_3D)[:]))
        xline_count = len(np.unique(segy.attributes(segyio.TraceField.CROSSLINE_3D)[:]))
        
        if inline_count > 1:
            slice_idx = inline_count // 2
            slice_data = cube[slice_idx]
            plt.title(f"Inline Slice (IL={slice_idx})")
        else:
            slice_idx = xline_count // 2
            slice_data = cube[:, slice_idx, :]
            plt.title(f"Crossline Slice (XL={slice_idx})")
        
        plt.imshow(slice_data.T, cmap='viridis', aspect='auto')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Trace Position')
        plt.ylabel('Depth (samples)')
        plt.savefig(save_path)
        plt.close()
        
    print(f"Visualization saved to: {save_path}")

if __name__ == "__main__":
    # Example usage
    npy_file = "data/data_decatur/processed/porosity_modeled_exploration_block.npy"
    template_segy = "data/data_decatur/seismic/porosity_cropped_final.segy"
    output_segy = "data/data_decatur/seismic/porosity_modeled_explo_new.segy"
    
    # Convert NPY to SEGY
    convert_npy_to_segy(
        npy_file_path=npy_file,
        output_segy_path=output_segy,
        template_segy_path=template_segy,
        start_depth=4700,
        sample_interval=20
    )