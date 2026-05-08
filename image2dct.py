# Comment
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fft import dctn

input_dir = Path("img50x50")
output_dir = Path("dct50x50")
output_dir.mkdir(exist_ok=True)

for i in range(1, 101):
    img_path = input_dir / f"image_{i:05d}.jpg"
    
    img = Image.open(img_path).convert("L")
    arr = np.asarray(img, dtype=np.float64)

    # 2D orthonormal DCT
    dct_arr = dctn(arr, type=2, norm="ortho")

    # Save numerical DCT coefficients
    np.save(output_dir / f"dct_{i:05d}.npy", dct_arr)

print(f"Saved 10000 DCT files to: {output_dir.resolve()}")


