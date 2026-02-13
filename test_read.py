import os
import numpy as np
import glob
import struct

workspace_path = "C:\\Users\\Anthony\\Documents\\Projects\\scene_agent\\breadth_agent\\results\\workspace"
depth_dir = os.path.join(workspace_path, "dense", "stereo", "depth_maps")
depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.photometric.bin")))

depth_maps = []

for file in depth_files:
    with open(file, "rb") as f:
        # Read until newline (end of the ASCII header)
        # ---- Read the text header ----
        header_bytes = b""
        count = 0
        while True:
            byte = f.read(1)
            if not byte:
                raise IOError(f"EOF before header finished")
            header_bytes += byte
            if byte == b"&":
                count += 1
                if count == 3:
                    # The header ends with an extra ampersand
                    # after channels, e.g. "1600&1200&1&"
                    break

        header_str = header_bytes.decode("ascii")
        parts = header_str.strip("&").split("&")
        print("HEADER", header_str)
        # Now read the numeric values that follow
        if len(parts) != 3:
            raise ValueError(f"Unexpected header format: {header_str}")

        width = int(parts[0])
        height = int(parts[1])
        channels = int(parts[2])
        # ---- Read the float32 depth data ----
        num_values = width * height
        depth_data = np.fromfile(f, dtype=np.float32, count=num_values)

        if depth_data.size != num_values:
            raise ValueError(
                f"Depth size mismatch, "
                f"expected {num_values}, got {depth_data.size}"
            )
        
        depth = depth_data.reshape((height, width))
        depth_maps.append(depth)
    print(depth_maps[0])
