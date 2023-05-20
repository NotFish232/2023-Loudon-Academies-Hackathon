from PIL import Image
import os

files = [f for f in os.listdir("imgs/") if f.endswith(".png")]

for i, filename in enumerate(files):
    f = Image.open(f'imgs/{filename}')  # My image is a 200x374 jpeg that is 102kb large
    f = f.resize((256, 256), Image.Resampling.LANCZOS)
    f.save(f"processed_imgs/pirate-{i}.png")

