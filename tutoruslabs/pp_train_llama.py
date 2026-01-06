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
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--micro_batch_size", type=int, default=4)
parser.add_argument("--pp_size", type=int, default=2)
parser.add_argument("--tp_size", type=int, default=2)
parser.add_argument("--dp_size", type=int, default=2)
parser.add_argument("--run_id", type=str)
parser.add_argument("--llama_access_token", type=str, required=True)
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

    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=access_token, use_cache=True)

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params


    if int(os.environ["RANK"]) == 0:
        print ('Total parameters in model: {:,}'.format(get_total_params(model)))


    batch_size = args.batch_size
    #micro_batch_size = int(os.environ["WORLD_SIZE"]) // 2 # TODO
    micro_batch_size = args.micro_batch_size

    if int(os.environ["RANK"]) == 0:
        print(f"total process count: {os.environ['WORLD_SIZE']}")
        print(f"batch size: {batch_size}")
        print(f"micro batch size: {micro_batch_size}")

    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, pp_size=4, tp_size=2, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, pp_size=2, dp_size=4, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, pp_size=args.pp_size, tp_size=args.tp_size, dp_size=args.dp_size, activation_ckpt=False, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=False, swap_model_in_optstep=False, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, pp_size=2, tp_size=4, dp_size=2, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, pp_size=4, tp_size=2, dp_size=2, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, tp_size=4, dp_size=4, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, tp_size=8, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, dp_size=4, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, tp_size=2, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    #optimus_p = Optimus_p(model, micro_batch_size, use_gpu=True, tp_size=4, activation_ckpt=True, force_free_mem=True, display_mem=True, swap_opt_in_fwdbwd=True, swap_model_in_optstep=True, ir_analyze=IR_Anal.SEQUENTIAL)
    print(f" rank={optimus_p.get_rank()} ...")

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


    # 1F1B 안전 가드
    num_mb = args.batch_size // args.micro_batch_size
    if num_mb < args.pp_size:
        print(f"ERROR: num_microbatches({num_mb}) < pp_size({args.pp_size}) → 1f1b 데드락 위험")
        EXIT_CODE = 60
        save_exit_code(EXIT_CODE, args.run_id)
        sys.exit(EXIT_CODE)


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