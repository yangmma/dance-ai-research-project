import json


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPL_JOINT_CHECK_MAPPINGS = {
    "left_collar": "left_shoulder",
    "right_collar": "right_shoulder",
    "left_shoulder": "left_elbow",
    "right_shoulder": "right_elbow",
    "left_elbow": "left_wrist",
    "right_elbow": "right_wrist",
    "left_hip": "left_knee",
    "right_hip": "right_knee",
    "left_knee": "left_ankle",
    "right_knee": "right_ankle",
}


def main(input_file_1: str, input_file_2: str):
    with open(input_file_1) as f:
        data_1 = json.loads(f.read())['dance_array']
    with open(input_file_2) as f:
        data_2 = json.loads(f.read())
    
    scale = []
    for a, b in zip(data_1, data_2):
        for from_joint in SMPL_JOINT_CHECK_MAPPINGS:
            to_joint = SMPL_JOINT_CHECK_MAPPINGS[from_joint]
            from_index = SMPL_JOINT_NAMES.index(from_joint)
            to_index = SMPL_JOINT_NAMES.index(to_joint)

            diff_1 = a[from_index] - a[to_index]
            diff_2 = b[from_index] - b[to_index]
            scale.append(diff_1 / diff_2)
        break
    
    print(scale)
    

if __name__ == "__main__":
    main("001.npy.json", "correct_corrected/dance_data_10_corrected.json")