from smplx import SMPL
import torch
import config.config as cf
from tqdm import tqdm
from utils.format import format_rotmat_output
import json
import os


if __name__ == "__main__":
    smpl = SMPL(model_path=cf.smpl_model_path, gender='MALE', batch_size=1).to(torch.device("cuda"))

    out_path = "./generated_out"
    processed_path = "./post_processed_out"
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    fnames = os.listdir(out_path)
    for name in tqdm(fnames):
        full_path = os.path.join(out_path, name)
        with open(full_path) as f:
            print(full_path)
            results = json.loads(f.read())

        result_outs = []
        for result in results:
            result = format_rotmat_output(result, smpl)
            result_outs.append(result)
        
        full_out_path = os.path.join(processed_path, name)
        with open(full_out_path, "w") as f:
            f.write(json.dumps(result_outs))
        os.remove(full_path)