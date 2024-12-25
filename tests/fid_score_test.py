import torch

from cleanfid import fid
print(torch.cuda.is_available())
fdir1='./resized_div2k_images/'
fdir2='resized_codeformer_results_w/resized_codeformer_results_w_0.9'
score = fid.compute_fid(fdir1, fdir2,mode="clean", num_workers=0)
print(score)