import numpy as np

tp_arr = [1,2,4]
model = ['llama']
gpu_type = ['A40']


for m in model:
    for g in gpu_type:   
        for tp in tp_arr:
            filename = f"gpt2XL_{gpu_type[0]}_{tp}_Llama-3.2-1B.npy"
            try:
                data_1 = np.load(filename)
            except FileNotFoundError:
                print(f"{filename} not found")
                continue
            print(f"{filename} len: {len(data_1)}")
            print(f"{data_1[0]:.8f} {data_1[1]:.8f} {data_1[-1]:.8f}")