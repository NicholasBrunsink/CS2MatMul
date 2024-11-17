import cerebras_pytorch as cb
from cerebras_pytorch import torch
import numpy as np
from time import perf_counter
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

device = torch.device(cb.device) if torch.cuda.is_available() else torch.device('cpu')


class TimingModel(torch.nn.Module):
    def __init__(self, matSize):
        super(TimingModel, self).__init__()
        self.lin = torch.nn.Linear(matSize,matSize, device=device, bias=False).half()
        self.lin.weight.requires_grad = False

    def forward(self, x):
        x = self.lin(x)
        return x

def testTime(model, data, times):
    timing = model.to(device)
    dataGPU = data.to(device)
    
    start = perf_counter()
    res = timing.forward(dataGPU)
    torch.cuda.synchronize(device)
    end = perf_counter()

    times.append(end-start)
    return res

def main():
    count = open("count.txt", "r")
    currCount = int(count.readline())
    count.close()
    out = open("results.txt", "a")
    matSizes = [1, 100, 1000, 5000, 10000, 15000]

    for size in matSizes:
        for sparsity in range(100, -1, -10):
            print(f"Running tests for mat size: {size} and sparsity: {sparsity}%")
            times = []
            numTrials = 10
            data = np.random.randint(1, 100, (size, size))
            data = torch.tensor(np.where(data<sparsity, 1, 0), dtype=torch.float16, requires_grad=False)
            model = TimingModel(size)
            for _ in range(numTrials+1):
                torch.cuda.empty_cache()
                res = testTime(model, data, times)
            finalTime = sum(times[1:])/numTrials
            print(f"\tTiming: {finalTime:.6f}s")
            out.write(f"{currCount},{size},{sparsity},{finalTime}\n")
        currCount+=1

    count = open("count.txt", "w")
    count.write(f"{currCount}")
    count.close()
    out.close()

main()