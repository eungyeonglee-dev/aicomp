#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
# Usage: torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> 
#                 --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama5.py <llama_access_token>
#
# *** This program was tested with torch 2.5.0 and transformers 4.46.2.
#     The version of transformers used must be consistent across all machines used for testing ***
#

import torch
import torch.nn as nn

import torch.distributed as dist
import datetime
import logging
import os
import sys
import math
import time
from packaging import version

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

import transformers

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from opt_prime.opti_pri import Optimus_p
from opt_prime.IR import IR_Anal

logging.basicConfig(level=logging.ERROR)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--micro_batch_size", type=int, default=4)
parser.add_argument("--pp_size", type=int, default=2)
parser.add_argument("--tp_size", type=int, default=2)
parser.add_argument("--dp_size", type=int, default=2)
parser.add_argument("--run_id", type=str)
parser.add_argument("--llama_access_token", type=str, required=True)
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=["train", "graph"],
    help='"graph": dump FX graph nodes with empty/meta weights (no training, no full checkpoint load).',
)
parser.add_argument(
    "--graph_out",
    type=str,
    default=None,
    help='Output path for graph dump (default: results/<model>_pp<PP>_fx_nodes.txt)',
)
parser.add_argument(
    "--include_node_args",
    action="store_true",
    help="Include node.args and all_input_nodes in dump (can be very large).",
)
args, unknown = parser.parse_known_args()

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)

RESULT_FILEPATH = os.path.join(RESULT_DIR, args.model_name.split("/")[1] + ".csv")
if not os.path.isfile(RESULT_FILEPATH):
    with open(RESULT_FILEPATH, "w", encoding="utf-8") as f:
        f.write("batch_size,micro_batch_size,pp_size,tp_size,dp_size,training_time(sec)\n")

def write_result(batch_size, micro_batch_size, pp_size, tp_size, dp_size, result, result_filepath):
    with open(result_filepath, "a", encoding="utf-8") as file:
        file.write(f"{batch_size},{micro_batch_size},{pp_size},{tp_size},{dp_size},{result}\n")

def save_exit_code(exit_code: int, run_id: str, elapsed_time: float | None = None):
    """
    rank 0 프로세스에서 EXIT_CODE를 /tmp 디렉터리에 기록합니다.
    EXIT_CODE가 0이고 elapsed_time이 주어지면 함께 기록합니다.
    다른 랭크에서는 아무 작업도 하지 않습니다.
    """
    try:
        # rank0만 기록
        if os.environ.get("RANK", "0") == "0":
            log_path = f"tmp/exitcode_{run_id}.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                if exit_code == 0 and elapsed_time is not None:
                    # "0,123.456" 형식으로 기록
                    f.write(f"{exit_code},{elapsed_time:.3f}")
                else:
                    # 그 외에는 exit_code만 기록
                    f.write(str(exit_code))
            print(f"[rank:0] EXIT_CODE {exit_code} saved to {log_path}")
    except Exception as e:
        print(f"[rank:0] Failed to save EXIT_CODE file: {e}")
        pass

EXIT_CODE=0
ELAPSED_TIME = None


###
rank = int(os.environ.get('RANK', '0'))
local_rank = int(os.environ.get('LOCAL_RANK', '0'))
world_size = int(os.environ.get('WORLD_SIZE', '1'))
local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', str(world_size)))
master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
master_port = os.getenv("MASTER_PORT", "29500")


init_method = "tcp://" + str(master_addr) + ":" + str(master_port)
print(f"rank:{rank}, world_size:{world_size}, init_method:{init_method}, local_world_size:{local_world_size}, local_rank:{local_rank}")

dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method)

group_gloo = dist.new_group(backend="gloo")
###

def _dump_fx_graph_only(model_name: str, access_token: str, pp_size: int):
    """Build model with empty weights and dump FX nodes after IR split (no checkpoint load)."""
    if rank != 0:
        # Only rank0 writes.
        return

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        from accelerate import init_empty_weights
    except Exception as e:
        raise RuntimeError(
            "graph mode requires `transformers` and `accelerate` installed in this environment."
        ) from e

    out = args.graph_out
    if out is None:
        base = model_name.split("/")[-1]
        out = os.path.join(RESULT_DIR, f"{base}_pp{pp_size}_fx_nodes.txt")

    cfg = AutoConfig.from_pretrained(model_name, token=access_token)
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    class _Tpl:
        def __init__(self):
            self.rank = 0
            self.num_stage = pp_size

    class _DummyOpt:
        def __init__(self):
            self.tpl = _Tpl()
            self.model2type = {"hf": 50, "sy": 51, "vt": 52}
        def is_first_stage(self):
            return True
        def get_rank(self):
            return 0

    ir = IR(model, _DummyOpt())
    ir.retrieve_IR(model)
    ir.split_IR(model, "simple", num_stage=pp_size)
    root = ir.model_ir[0]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"pp_size={pp_size}\n")
        f.write("---- FX nodes ----\n")
        for n in root.graph.nodes:
            if args.include_node_args:
                f.write(
                    f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}, n.args:{n.args}, n.all_input_nodes:{n.all_input_nodes}\n"
                )
            else:
                f.write(f"n.op:{n.op}, n.name:{n.name}, n.target:{n.target}\n")
        f.write("---- submodules ----\n")
        for name, m in root.named_children():
            f.write(f"{name}: {m.__class__.__name__}\n")

    print(f"[rank:0][graph] wrote: {out}")


try:
    access_token = args.llama_access_token
    if access_token is None:
        raise ValueError("LLAMA_ACCESS_TOKEN environment variable is not set."
                        "       [Usage:] torchrun --nproc_per_node=<#_of_GPUs_per_node> --nnodes=<#_of_nodes> --node_rank=<current_node_rank> --master_addr=<IP_of_rank_0> --master_port=29500 pp_train_llama.py <llama_access_token>")


    #
    # This program needs torch version 2.3.1 or higher !!!
    #
    required_version = "2.3.1"
    current_version = torch.__version__

    if version.parse(current_version) >= version.parse(required_version):
        print(f"[rank:{int(os.environ['RANK'])}] torch version 2.3.1 or higher --> OK")
    else:
        print(f"[rank:{int(os.environ['RANK'])}] current torch version is {current_version}.")
        raise ValueError('This program needs torch version 2.3.1 or higher.')

    #
    # This program needs transformers version 4.46.2 or higher !!!
    #
    required_tf_version = "4.46.2"
    current_tf_version = transformers.__version__

    if version.parse(current_tf_version) >= version.parse(required_tf_version):
        print(f"[rank:{int(os.environ['RANK'])}] transformers version 4.46.2 or higher --> OK")
    else:
        print(f"[rank:{int(os.environ['RANK'])}] current transformers version is {current_tf_version}.")
        raise ValueError('This program needs transformers version 4.46.2 or higher.')


    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params


    ###

    batch_size = args.batch_size
    #micro_batch_size = int(os.environ["WORLD_SIZE"]) // 2 # TODO
    micro_batch_size = args.micro_batch_size

    if args.mode == "graph":
        _dump_fx_graph_only(args.model_name, access_token, args.pp_size)
        EXIT_CODE = 0
        save_exit_code(EXIT_CODE, args.run_id or "graph", None)
        sys.exit(EXIT_CODE)

    
    for i in range(local_world_size):
        if local_rank == i:
            model = AutoModelForCausalLM.from_pretrained(args.model_name, token=access_token, use_cache=False)

            if int(os.environ["RANK"]) == 0:
                print('Total parameters in model: {:,}'.format(get_total_params(model)))

            #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, tp_size=2, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.PARALLEL) ## IR_Anal.PARALLEL
            optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, pp_size=args.pp_size, tp_size=args.tp_size, dp_size=args.dp_size, activation_ckpt=False, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=IR_Anal.PARALLEL, pre_barrier=group_gloo) ## IR_Anal.PARALLEL
            #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, dp_size=2, activation_ckpt=False, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.PARALLEL, pre_barrier=group_gloo) ## IR_Anal.PARALLEL
            print(f" rank={optimus_p.get_rank()} ...")

        if local_rank > i:
            print(f"..[local_rank:{local_rank}, i:{i}] Before barrier()...")
            dist.barrier(group=group_gloo)
            print(f"..[local_rank:{local_rank}, i:{i}] After barrier()...................................")
    ###


    # TODO
    #get_info(optimus_p)

    optimus_p.train()

    optimus_p.optimizer = torch.optim.Adam(optimus_p.parameters(), lr=3e-5, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimus_p.optimizer, 1.0, gamma=0.95)

    datasets = load_dataset("squad").data["train"]["context"]
    datasets = [str(record) for record in datasets if len(str(record)) < 500]
    #dataloader = DataLoader(datasets, batch_size=batch_size, num_workers=4)
    dataloader = optimus_p.prepare_dataloader(datasets, batch_size)
    data_size=len(dataloader.dataset)
    print(f"data_size={data_size}")
    nbatches = len(dataloader)
    print(f"nbatches={nbatches}")

    """
    # 1F1B 안전 가드
    num_mb = args.batch_size // args.micro_batch_size
    if num_mb < args.pp_size:
        print(f"ERROR: num_microbatches({num_mb}) < pp_size({args.pp_size}) → 1f1b 데드락 위험")
        EXIT_CODE = 60
        save_exit_code(EXIT_CODE, args.run_id)
        sys.exit(EXIT_CODE)
    """

    epochs = 1 # The number of epochs

    def train():

        optimus_p.train() # turn on the train mode

        total_loss = 0
        start_time = time.time()


        for i, batch in enumerate(dataloader):

            data, labels = None, None

            # prepare input and label
            if optimus_p.is_first_stage():
                tokens = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt")
                data, labels = tokens.input_ids, tokens.input_ids

            labels = optimus_p.move_labels2last_stage(labels)

            optimus_p.optimizer.zero_grad()

            print(">>> OPTIMUS RUN\n")
            #optimus_p.run(data, labels)
            #optimus_p.run(data, labels, mode="gpipe")
            optimus_p.run(data, labels, mode="1f1b")

            if optimus_p.is_last_stage():
                loss = optimus_p.get_loss() 
            else:
                loss = None

            print(">>> OPTIMUS STEP\n")
            optimus_p.optimizer.step()

            if optimus_p.is_last_stage():
                loss = sum(loss) / optimus_p.mbsize
                total_loss += loss
                log_interval = 10
                if i % log_interval == 0 and i > 0:
                    cur_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | '
                        'lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                            epoch, i, nbatches, scheduler.get_lr()[0],
                            elapsed * 1000 / log_interval,
                            cur_loss, math.exp(cur_loss)))

                    total_loss = 0
                    start_time = time.time()

    if optimus_p.get_rank() == 0:
        tick = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        scheduler.step()

    if optimus_p.get_rank() == 0:
        tock = time.time()
        ELAPSED_TIME = tock - tick

        print('Time elapsed: %.3f sec ' % (ELAPSED_TIME))
        #write_result(args.batch_size, args.micro_batch_size, args.pp_size, args.tp_size, args.dp_size, f"{elapsed_time:.3f}", RESULT_FILEPATH)

        EXIT_CODE = 0

    """
    if dist.is_initialized():
        try:
            dist.barrier()
            print(f"[rank:{optimus_p.get_rank()} >> barrier ...")
            torch.cuda.synchronize()
            print(f"[rank:{optimus_p.get_rank()} >> synchronize...")
            dist.destroy_process_group()
        except Exception as e:
            print(f"Cleanp on rank {optimus_p.get_rank()}: {e}")
    
        print(f"[rank:{optimus_p.get_rank()}, run completed ...")
    """
    

except torch.cuda.OutOfMemoryError as e:
    print(f"ERROR: Out of GPU memory. {e}")
    #write_result(args.batch_size, args.micro_batch_size, args.pp_size, args.tp_size, args.dp_size, "OOM ERROR", RESULT_FILEPATH)
    EXIT_CODE = 10

except dist.DistBackendError as dbe:
    print(f"ERROR: Distributed communication failed. {dbe}")
    #write_result(args.batch_size, args.micro_batch_size, args.pp_size, args.tp_size, args.dp_size, "DIST ERROR", RESULT_FILEPATH)
    EXIT_CODE = 20

except Exception as e:
    print(f"ERROR: Unexpected error. {e}")
    #write_result(args.batch_size, args.micro_batch_size, args.pp_size, args.tp_size, args.dp_size, "EXCEPTION", RESULT_FILEPATH)
    EXIT_CODE = 30

finally:
    # dist.get_rank() 호출 금지 (PG 없을 수 있음)
    try:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
                print(f"[rank:{os.environ.get('RANK','?')}] process group destroyed.")
            except Exception as e:
                print(f"[rank:{os.environ.get('RANK','?')}] destroy_process_group failed: {e}")
                if EXIT_CODE == 0:
                    EXIT_CODE = 40
        else:
            # PG가 없으면 아무 것도 하지 않음
            pass
    except Exception as e:
        print(e)
        if EXIT_CODE == 0:
            EXIT_CODE = 41


print(">>> EXIT_CODE: ", EXIT_CODE, ELAPSED_TIME)
save_exit_code(EXIT_CODE, args.run_id, ELAPSED_TIME)
sys.exit(EXIT_CODE)