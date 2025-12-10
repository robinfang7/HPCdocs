import os
import torch
import torch.distributed as dist
import argparse

def main():
    # 1. 初始化分佈式環境
    # PyTorch 建議使用 'env://' 初始化，它會從環境變數 (如 Slurm 設置的) 自動獲取所需資訊。
    dist.init_process_group(backend="nccl")

    # 2. 獲取並輸出關鍵的分佈式資訊

    # 總體 (Global) Rank: 程式在所有節點/所有 GPU 中的唯一 ID (從 0 到 N-1)
    global_rank = dist.get_rank()

    # 世界大小 (World Size): 參與訓練的總進程/GPU 數量
    world_size = dist.get_world_size()

    # 節點 (Local) Rank: 程式在當前節點 (主機) 內的唯一 ID (從 0 到 G-1, G 為該節點 GPU 數)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # 當前節點的名稱 (Hostname)
    node_name = os.environ.get("SLURMD_NODENAME", "Unknown Node")

    # 獲取並設置當前進程應該使用的 GPU 設備
    if torch.cuda.is_available():
        # 設置設備為該進程的 local_rank 對應的 GPU
        # PyTorch DDP 和 Slurm/Torchrun 會確保 local_rank 正確對應到可用的 GPU
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        gpu_name = "N/A"

    # 輸出所需資訊
    print(f"--- Process Info ---")
    print(f"Node Name: {node_name}")
    print(f"Global Rank: {global_rank} / {world_size}")
    print(f"Local Rank (GPU Index on Node): {local_rank}")
    print(f"Using Device: {device} ({gpu_name})")

    # 3. 模擬訓練步驟 (可替換為您的實際模型和數據集)
    if global_rank == 0:
        print("\nStarting a mock training loop...")

    # 創建一個小型模型並將其移動到正確的設備
    model = torch.nn.Linear(10, 1).to(device)
    # 將模型包裹在 DDP 中
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # 模擬數據和訓練步驟
    data = torch.randn(2, 10).to(device)
    target = torch.randn(2, 1).to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    # 模擬一個訓練步驟
    output = ddp_model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    # 確保所有進程都完成後再退出
    dist.barrier()

    if global_rank == 0:
        print("Mock training complete. All processes synchronized.")

    # 4. 清理分佈式環境
    dist.destroy_process_group()

if __name__ == "__main__":
    main()