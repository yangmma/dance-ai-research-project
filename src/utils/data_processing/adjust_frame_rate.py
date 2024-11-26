import os
import json
import argparse
from tqdm import tqdm


def main(input_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fnames = os.listdir(input_dir)
    for name in tqdm(fnames):
        path = os.path.join(input_dir, name)
        if os.path.isdir(path):
            continue

        with open(path) as f:
            json_data = json.loads(f.read())
        
        frames = []
        for i, frame in enumerate(json_data):
            if i % 2 == 0:
                frames.append(frame)
            
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            f.write(json.dumps(frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    main(args.input, args.output)