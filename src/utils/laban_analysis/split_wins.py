import os
import numpy as np
from tqdm import tqdm
import json


def main():
    input_dir = "post_processed_out_split"
    for name in tqdm(os.listdir(input_dir)):
        full_path = os.path.join(input_dir, name)
        with open(full_path) as f:
            result = json.loads(f.read())
            result = np.array(result)
        
        t, j = result.shape
        split_c = t // 224
        chop = -(t % split_c) if t % split_c != 0 else len(result)
        result = result[: chop]
        results = np.split(result, split_c)
        for i, res in enumerate(results):
            full_out_path = os.path.join("./post_processed_out_split_win", f"{name}-{i}.json")
            with open(full_out_path, "w") as f:
                f.write(json.dumps(res.tolist()))


if __name__ == "__main__":
    main()