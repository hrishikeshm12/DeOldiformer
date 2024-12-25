import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(torch.cuda.is_available())  # Should return True
        print(torch.cuda.device_count())  # Should return the number of available GPUs
        print(torch.cuda.current_device())  # Should return the current device index (usually 0)
        print(torch.cuda.get_device_name(0))  # Should return the name of the first GPU

    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    check_cuda()
