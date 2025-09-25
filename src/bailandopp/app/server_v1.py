from sanic import Sanic, HTTPResponse
from easydict import EasyDict
import json
import asyncio

from utils.extractor import FeatureExtractor
import numpy as np
from essentia.standard import *
from smplx import SMPL
import os
import torch

from models_v1.bailando_v1 import BailandoV1
import config.config as cf
import config.gpt_config_lb as gpt_cf
import config.vqvae_config_lb as vq_cf
from utils.format import format_output
from sanic.worker.manager import WorkerManager

# global
WorkerManager.THRESHOLD = 600 # Value is in 0.1s
DEFAULT_SAMPLING_RATE = 15360*2
DEFAULT_MUSIC_PATH = "../test_client/data/music/jazz_ballet.wav"
DEFAULT_MUSIC_ID = "jazz_ballet"
DEFAULT_SAVE_DIR = "./data"
app = Sanic("ai_agent_server")
app.ctx.fusion_events = {}

# boostrap
@app.before_server_start
async def boostrap(app, loop):
    print("[BOOSTRAP] Initializing..")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"use device: {device}")
    print(f"[BOOTSTRAP] Initializing AI Agent Model")
    app.ctx.agent = BailandoV1(vq_cf, gpt_cf, cf, device, "./weight/vqvae_lb.pt", "./weight/gpt_lb.pt")
    print(f"[BOOTSTRAP] Initializing SMPL Model")
    app.ctx.smpl = SMPL(model_path=cf.smpl_model_path, gender='MALE', batch_size=1).to(torch.device(device))
    #app.ctx.prev = []
    #app.ctx.index = 0
    app.ctx.state_by_participant = {}
    app.ctx.fusion_buffer = {}
    app.ctx.fusion_refcount = {}
    print("[BOOSTRAP] Complete")


@app.post("/dance-sequence")
async def generate_dance_sequence(request):
    print("received generate dance sequence request")
    request = EasyDict(request.json)
    game_mode = request.get("gameMode", 0) # 0: singleplayer, 1: two agents, 2: one agent (integrated)
    participant_id = request.get("participantID", "unknown")
    print(f"handling request for participnt: {participant_id}, gameMode: {game_mode}")

    if participant_id not in app.ctx.state_by_participant:
        app.ctx.state_by_participant[participant_id] = {
            "prev": [],
            "index": 0,
            "pending_payloads": {} # create a dict buffer to ensure both payloads are received before fusing
        }

    state = app.ctx.state_by_participant[participant_id]

    startFrameIndex = request.startFrameIndex
    payload = request.payload
    length = request.length # how long of a clip to generate.
    shift = request.shift # amount of seed from previous motion clip to take.
    seed = request.seed # amount of user input to generate from, this will override user input from pos 0.

    # save payload for analysis
    file = f"dance_{state['index']}_{participant_id}"
    path = os.path.join(DEFAULT_SAVE_DIR, file)
    with open(path, "w") as f:
        f.write(json.dumps(payload))

    if game_mode in [0, 1]:
        result, quant = await handle_generate_dance_sequence(start_frame_index=startFrameIndex, payload=payload, length=length, shift=shift, seed=seed, prev_sequence=state["prev"])
        result = result.squeeze(0).cpu().numpy().tolist()
        result = format_output(result)
        print(np.shape(result))

        #app.ctx.prev = result
        state["prev"] = result
        state["index"] += 1

        quant_up, quant_down = quant
        quant = [quant_up.tolist(), quant_down.tolist()]

        response = {
            'result': result,
            'quant': quant,
            'participantID': participant_id
        }
        response = json.dumps(response)
        print(f"CHECK completed generate dance sequence request for {game_mode}")
        return HTTPResponse(body=response, status=200)
    
    elif game_mode == 2:
        print(f"Gamemode {game_mode} found, fusing input streams...")
        
        # Ensure fusion buffer and events exist
        if startFrameIndex not in app.ctx.fusion_buffer:
            app.ctx.fusion_buffer[startFrameIndex] = {}
            app.ctx.fusion_events[startFrameIndex] = asyncio.Event()

        app.ctx.fusion_buffer[startFrameIndex][participant_id] = payload
        # Debug: show who is currently in the shared buffer for this frame
        print(f"[DEBUG] Fusion buffer for frame {startFrameIndex}: {list(app.ctx.fusion_buffer[startFrameIndex].keys())}")

        # required participants list
        required_participants = ["0", "1"]
        participant_id = str(request.get("participantID", "unknown"))
        print("Participant id is:", participant_id, type(participant_id))
        
        # check if all payloads are present:
        if all(pid in app.ctx.fusion_buffer[startFrameIndex] for pid in required_participants):
            app.ctx.fusion_events[startFrameIndex].set()
        else:
            print(f"[DEBUG] Waiting for other participant to send payload.")
            try:
                await asyncio.wait_for(app.ctx.fusion_events[startFrameIndex].wait(), timeout=5.0)  # optional timeout
            except asyncio.TimeoutError:
                print(f"[DEBUG] Asyncio timed out! Returning message to the frontend...")
                last_fused = app.ctx.state_by_participant[participant_id]["prev"]
                return HTTPResponse(
                    body=json.dumps({
                        "status": "timeout_waiting_for_other",
                        "result": last_fused,
                        "participantID": participant_id
                    }),
                    status=200
                )

        # both payloads present -> fuse!
        print("[DEBUG] Both payloads present; performing quant-level fusion.")
        agent: BailandoV1 = app.ctx.agent
        payload0 = app.ctx.fusion_buffer[startFrameIndex]["0"]
        payload1 = app.ctx.fusion_buffer[startFrameIndex]["1"]

        # Convert to tensors and encode
        tensor0 = torch.tensor(np.array(payload0)).unsqueeze(0).to(agent.device)
        tensor1 = torch.tensor(np.array(payload1)).unsqueeze(0).to(agent.device)
        print(f"[DEBUG] tensor0 shape: {tensor0.shape}, tensor1 shape: {tensor1.shape}")

        quants0 = agent.vqvae.module.encode(tensor0)
        quants1 = agent.vqvae.module.encode(tensor1)

        if isinstance(quants0, tuple):
            fused_quants = tuple(((quants0[i][0] + quants1[i][0]) / 2.0).round().long() for i in range(len(quants0)))
        else:
            fused_quants = ((quants0[0] + quants1[0]) / 2.0).round().long()
        print("Fused quants!")

        # generate fused dance from fused quants
        zs = agent.gpt.module.sample(fused_quants, shift=shift, length=length)
        fused_pose = agent.vqvae.module.decode(zs)
        fused_pose = fused_pose.squeeze(0).detach().cpu().numpy().tolist()
        fused_pose = format_output(fused_pose)
        print(f"Fused pose shape: {np.shape(fused_pose)}")

        # Update prev state for both participants
        for pid in required_participants:
            if pid in app.ctx.state_by_participant:
                app.ctx.state_by_participant[pid]["prev"] = fused_pose
                app.ctx.state_by_participant[pid]["index"] += 1

        # clean up pending payloads from the shared buffer
        if startFrameIndex not in app.ctx.fusion_refcount:
            app.ctx.fusion_refcount[startFrameIndex] = len(required_participants)

        app.ctx.fusion_refcount[startFrameIndex] -= 1
        if app.ctx.fusion_refcount[startFrameIndex] == 0:
            del app.ctx.fusion_buffer[startFrameIndex]
            del app.ctx.fusion_events[startFrameIndex]
            del app.ctx.fusion_refcount[startFrameIndex]
            print(f"Deleted fusion buffer and events for frame {startFrameIndex}.")

        print(f"[DEBUG] zs type: {type(zs)}")
        print(f"[DEBUG] zs value: {zs}")

        def tensor_to_list(obj):
            print("Attempting to convert to list...")
            all_rows = []
            if isinstance(zs, tuple):
                for lst in zs:  # lst is usually a list with one tensor
                    for t in lst:
                        if torch.is_tensor(t):
                            all_rows.extend(t.detach().cpu().tolist())
                        else:
                            all_rows.append(t)
            elif isinstance(zs, list):
                for t in zs:
                    if torch.is_tensor(t):
                        all_rows.extend(t.detach().cpu().tolist())
                    else:
                        all_rows.append(t)
            elif torch.is_tensor(zs):
                all_rows = zs.detach().cpu().tolist()
            else:
                all_rows = [[zs]]
            return all_rows

        quant_list = tensor_to_list(zs)

        response = {
            'result': fused_pose,
            'quant': quant_list,
            'participantID': participant_id
        }
        print(f"Returning fused dance to participant {participant_id}")

        response_json = json.dumps(response)
        print(f"Completed fusion for frame {startFrameIndex}, participant {participant_id}. Returning HTTPResponse.")
        return HTTPResponse(body=response_json, status=200)



async def handle_generate_dance_sequence(start_frame_index, payload, length, shift, seed, prev_sequence):
    print("handling generate dance sequence request")
    agent: BailandoV1 = app.ctx.agent

    # transform
    np_dance = np.array(payload)
    print(np.shape(np_dance))
    if seed > 0 and prev_sequence is not None and len(prev_sequence) >= seed:
        print(f"using seed motion; count: {seed}")
        input_seed = np.array(prev_sequence[:seed])
        np_dance = np.concatenate((input_seed, np_dance), axis=0)
    print(np.shape(np_dance))
    root = np_dance[:, :3]
    np_dance = np_dance - np.tile(root, (1, 24))
    np_dance[:, :3] = root
    for kk in range((len(np_dance) // 5 + 1) * 5 - len(np_dance) ):
        np_dance = np.append(np_dance, np_dance[-1:], axis=0)
    dance_input = torch.tensor(np_dance).unsqueeze(0)

    result, quants = agent.eval_raw(dance_input, length, start_frame_index, shift)
    return result, quants
