# 1. grant script permissions
chmod +x run_send-recv-tensor.sh

# 2. copy to container (modify path as needed)
sudo docker cp run_send-recv-tensor.sh optimus-prime:/workspace/aicomp/opt_prime/

# 3. execute script with Docker exec
sudo docker exec optimus-prime bash -lc "/workspace/aicomp/opt_prime/run_send-recv-tensor.sh"