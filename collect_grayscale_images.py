import requests
from pathlib import Path

out = Path("grayscale_100x100_images_test")
out.mkdir(exist_ok=True)

for i in range(10001, 10101):
    url = f"https://picsum.photos/seed/random_{i}/100/100?grayscale"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    (out / f"image_{i:05d}.jpg").write_bytes(r.content)

print("Done.")