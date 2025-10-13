from sanic import Sanic, HTTPResponse
from easydict import EasyDict
import json
import asyncio
import random

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
    app.ctx.fusion_turn = 0
    print("[BOOSTRAP] Complete")


@app.post("/dance-sequence")
async def generate_dance_sequence(request):
    """
    Generates dance sequence based on input payload and game mode, set three modes: 0 for singleplayer, 1 for two agents, 2 for one agent (co-created response)
    """
    print("received generate dance sequence request")
    request = EasyDict(request.json)
    game_mode = request.get("gameMode", 0)
    participant_id = request.get("participantID", "unknown")
    print(f"handling request for participant: {participant_id}, gameMode: {game_mode}")

    if participant_id not in app.ctx.state_by_participant:
        app.ctx.state_by_participant[participant_id] = {
            "prev": [],
            "index": 0,
            "pending_payloads": {}
        }

    state = app.ctx.state_by_participant[participant_id]

    startFrameIndex = request.startFrameIndex
    payload = request.payload
    length = request.length # how long of a clip to generate
    shift = request.shift # amount of seed from previous motion clip to take
    seed = request.seed # amount of user input to generate from, this will override user input from pos 0

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
    
    elif game_mode == 2: # single agent, co-creation
        print(f"Gamemode {game_mode} found, fusing input streams...")
        
        # ensure fusion buffer and events exist
        if startFrameIndex not in app.ctx.fusion_buffer:
            app.ctx.fusion_buffer[startFrameIndex] = {}
            app.ctx.fusion_events[startFrameIndex] = asyncio.Event()

        app.ctx.fusion_buffer[startFrameIndex][participant_id] = payload
        print(f"[DEBUG] Fusion buffer for frame {startFrameIndex}: {list(app.ctx.fusion_buffer[startFrameIndex].keys())}")
        
        required_participants = ["0", "1"]
        participant_id = str(request.get("participantID", "unknown"))
        print("Participant id is:", participant_id, type(participant_id))

        # synchronization init
        app.ctx.fusion_events.setdefault(startFrameIndex, asyncio.Event())
        app.ctx.fusion_locks = getattr(app.ctx, "fusion_locks", {})
        app.ctx.fusion_results = getattr(app.ctx, "fusion_results", {})
        app.ctx.fusion_result_events = getattr(app.ctx, "fusion_result_events", {})
        app.ctx.fusion_locks.setdefault(startFrameIndex, asyncio.Lock())
        app.ctx.fusion_results.setdefault(startFrameIndex, None)
        app.ctx.fusion_result_events.setdefault(startFrameIndex, asyncio.Event())

        if all(pid in app.ctx.fusion_buffer[startFrameIndex] for pid in required_participants):
            app.ctx.fusion_events[startFrameIndex].set()

        # check if all payloads are present:
        if not all(pid in app.ctx.fusion_buffer[startFrameIndex] for pid in required_participants):
            print(f"[DEBUG] Waiting for other participant to send payload.")
            try:
                await asyncio.wait_for(app.ctx.fusion_events[startFrameIndex].wait(), timeout=5.0)
            except asyncio.TimeoutError:
                print(f"[DEBUG] Asyncio timed out! Returning message to the frontend...")
                last_fused = app.ctx.state_by_participant[participant_id]["prev"]
                return HTTPResponse(
                    body=json.dumps({
                        "status": "timeout_waiting_for_other",
                        "result": last_fused,
                        "quant": [],
                        "participantID": participant_id
                    }),
                    status=200
                )

        # BOTH payloads are present now. Use lock+result-event to ensure only ONE fusion happens (otherwise causes bug)
        lock = app.ctx.fusion_locks[startFrameIndex]
        result_event = app.ctx.fusion_result_events[startFrameIndex]

        if not result_event.is_set():
            async with lock:
                if not result_event.is_set():
                    try:
                        print("[DEBUG] Leader acquired lock — performing quant-level fusion.")
                        agent: BailandoV1 = app.ctx.agent
                        payload0 = app.ctx.fusion_buffer[startFrameIndex]["0"]
                        payload1 = app.ctx.fusion_buffer[startFrameIndex]["1"]

                        tensor0 = torch.tensor(np.array(payload0)).unsqueeze(0).to(agent.device)
                        tensor1 = torch.tensor(np.array(payload1)).unsqueeze(0).to(agent.device)

                        quants0 = agent.vqvae.module.encode(tensor0)
                        quants1 = agent.vqvae.module.encode(tensor1)

                        # choose weight and fuse
                        if app.ctx.fusion_turn % 2 == 0:
                            weight_p0 = 0.7
                        else:
                            weight_p0 = 0.3
                        fused_quants = fuse_quants_weighted_random(quants0, quants1, block_size=16, weight_p0=weight_p0) # function can be changed if needed
                        app.ctx.fusion_turn += 1

                        zs = agent.gpt.module.sample(fused_quants, shift=shift, length=length)
                        fused_pose = agent.vqvae.module.decode(zs)
                        fused_pose = fused_pose.squeeze(0).detach().cpu().numpy().tolist()
                        fused_pose = format_output(fused_pose)

                        quant_list = None

                        def tensor_to_list_local(obj):
                            """
                            Data conversion to Python list for response
                            """
                            all_rows = []
                            if isinstance(obj, tuple):
                                for lst in obj:
                                    for t in lst:
                                        if torch.is_tensor(t):
                                            all_rows.extend(t.detach().cpu().tolist())
                                        else:
                                            all_rows.append(t)
                            elif isinstance(obj, list):
                                for t in obj:
                                    if torch.is_tensor(t):
                                        all_rows.extend(t.detach().cpu().tolist())
                                    else:
                                        all_rows.append(t)
                            elif torch.is_tensor(obj):
                                all_rows = obj.detach().cpu().tolist()
                            else:
                                all_rows = [[obj]]
                            return all_rows
                        quant_list = tensor_to_list_local(zs)

                        # prepare response object once; store the JSON so followers return exactly same payload
                        response_obj = {
                            "result": fused_pose,
                            "quant": quant_list,
                        }
                        app.ctx.fusion_results[startFrameIndex] = response_obj

                        for pid in required_participants:
                            if pid in app.ctx.state_by_participant:
                                app.ctx.state_by_participant[pid]["prev"] = fused_pose
                                app.ctx.state_by_participant[pid]["index"] += 1

                        result_event.set()
                        print("[DEBUG] Leader finished fusion and set result_event.")
                    except Exception as e:
                        # ensure followers aren't left waiting forever
                        print(f"[ERROR] Fusion leader failed: {e}", flush=True)
                        app.ctx.fusion_results[startFrameIndex] = {
                            "status": "error",
                            "error": str(e)
                        }
                        result_event.set()
                        raise

        else:
            print("[DEBUG] Follower detected result ready; will wait for event/return stored response.")

        # Wait for the result_event (if leader, event already set)
        await result_event.wait()
        stored_response = app.ctx.fusion_results[startFrameIndex]

        response = {
            "result": stored_response.get("result"),
            "quant": stored_response.get("quant"),
            "participantID": participant_id
        }

        # clean up
        app.ctx.fusion_refcount.setdefault(startFrameIndex, len(required_participants))
        app.ctx.fusion_refcount[startFrameIndex] -= 1
        if app.ctx.fusion_refcount[startFrameIndex] == 0:
            del app.ctx.fusion_buffer[startFrameIndex]
            del app.ctx.fusion_locks[startFrameIndex]
            del app.ctx.fusion_results[startFrameIndex]
            del app.ctx.fusion_result_events[startFrameIndex]
            del app.ctx.fusion_refcount[startFrameIndex]
            if startFrameIndex in app.ctx.fusion_events:
                del app.ctx.fusion_events[startFrameIndex]
            print(f"Deleted fusion buffer and events for frame {startFrameIndex}.")

        response_json = json.dumps(response)
        print(f"Returning fused dance to participant {participant_id}")
        return HTTPResponse(body=response_json, status=200)


def fuse_quants_average(quants0, quants1):
    """
    Simple averaging of two sets of quants (one from each participant)
    """
    if isinstance(quants0, tuple):
        fused_quants = tuple(((quants0[i][0] + quants1[i][0]) / 2.0).round().long() for i in range(len(quants0)))
    else:
        fused_quants = ((quants0[0] + quants1[0]) / 2.0).round().long()
    print("Fused quants by average!")
    return fused_quants


def fuse_quants_alternating(quants0, quants1, block_size=16):
    """
    Alternates blocks between both participants
    """
    if isinstance(quants0, tuple):
        fused_quants = []
        for i in range(len(quants0)):
            blocks = []
            total_len = quants0[i][0].shape[0]
            for j in range(0, total_len, block_size):
                if (j // block_size) % 2 == 0:
                    blocks.append(quants0[i][0][j:j + block_size])
                else:
                    blocks.append(quants1[i][0][j:j + block_size])
            fused_quants.append(torch.cat(blocks, dim = 0))
        fused_quants = tuple(fused_quants)
    else:
        blocks = []
        total_len = quants0[0].shape[0]
        for j in range(0, total_len, block_size):
            if (j // block_size) % 2 == 0:
                blocks.append(quants0[0][j:j+block_size])
            else:
                blocks.append(quants1[0][j:j+block_size])
        fused_quants = torch.cat(blocks, dim=0)
    print("Fused quants with alternating strategy!")
    return fused_quants


def fuse_quants_weighted_random(quants0, quants1, block_size=16, weight_p0=0.7):
    """
    Randomly chooses blocks from participant 0 or 1's quantized motion sequences according to weight_p0
    """
    print(f"Current weight is {weight_p0} for p0.")
    if isinstance(quants0, tuple):
        fused_quants = []
        for i in range(len(quants0)):
            blocks = []
            q0 = quants0[i][0]
            q1 = quants1[i][0]
            total_len = min(q0.shape[0], q1.shape[0])
            for j in range(0, total_len, block_size):
                if random.random() < weight_p0:
                    chosen = q0[j:j + block_size]
                    source = 0
                else:
                    chosen = q1[j:j+block_size]
                    source = 1
                print(f"[DEBUG] Block {j//block_size}: chose participant {source}, frames {j}-{j+block_size}")
                blocks.append(chosen)
            fused_quants.append(torch.cat(blocks, dim=0))
        fused_quants = tuple(fused_quants)
    else:
        # Single participant fallback CAN U REVISIT IF THIS FALLBACK IS NECESSARY, COULD BE CLEANED
        q0 = quants0[0]
        q1 = quants1[0]
        total_len = min(q0.shape[0], q1.shape[0])
        blocks = []

        for j in range(0, total_len, block_size):
            if random.random() < weight_p0:
                chosen = q0[j:j+block_size]
                source = 0
            else:
                chosen = q1[j:j+block_size]
                source = 1
            print(f"[DEBUG] Block {j//block_size}: chose participant {source}, frames {j}-{j+block_size}")
            blocks.append(chosen)
        fused_quants = torch.cat(blocks, dim=0)
    print("Fused quants with weighted random turn-taking!")
    return fused_quants


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
