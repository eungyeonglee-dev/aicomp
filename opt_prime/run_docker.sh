cd ${HOME}/workspace && docker run -d --name optimus-timelog --gpus all --ipc=host \
--network=host -v ${HOME}/workspace/aicomp:/workspace/aicomp \
-w /workspace/aicomp \
-e LLAMA_ACCESS_TOKEN="$LLAMA_ACCESS_TOKEN" optimus-timelog bash -lc "tail -f /dev/null"