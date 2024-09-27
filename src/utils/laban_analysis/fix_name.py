import os

fnames = os.listdir("./new_clips")
tmp_name_map = {}
for name in fnames:
    split_name = name.split("_")
    new_name_index = int(split_name[2]) + 101
    split_name[2] = str(new_name_index)
    new_name = "_".join(split_name[1:])
    new_name = f"tmp_{new_name}"
    # new_name = "dance_" + name
    print(new_name)
    # full_path = os.path.join("./new_clips", name)
    # tmp_path = os.path.join("./new_clips", new_name)
    # os.rename(full_path, tmp_path)
