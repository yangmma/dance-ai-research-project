import numpy as np
import config.config as cf
from tqdm import tqdm
import json
import os

if __name__ == "__main__":
    out_path = "./post_processed_out"
    processed_path = "./post_processed_out_split"
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    fnames = os.listdir(out_path)
    for name in tqdm(fnames):
        full_path = os.path.join(out_path, name)
        with open(full_path) as f:
            results = json.loads(f.read())
        
        for i, result in enumerate(results):
            full_out_path = os.path.join(processed_path, f"{name}-{i}.json")
            with open(full_out_path, "w") as f:
                f.write(json.dumps(result))