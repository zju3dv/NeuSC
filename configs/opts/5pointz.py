# define uninterested regions by meshlab
import torch

def return_interested(xyz):
    # default all True
    # return torch.ones_like(xyz[..., 1]).bool()
    
    # filter ground
    xyz_mask = xyz[..., 1] <= 0.35 # filter ground

    # filter center noisy regions of 5pointz
    # mannually selected from meshlab
    un = torch.logical_and(xyz[..., 0]>-2.75676727, xyz[..., 0]<1.13193512)
    un = torch.logical_and(un, torch.logical_and(xyz[..., 2]>0.266700, xyz[..., 2]<4.53462887))
    un = torch.logical_and(un, torch.logical_and(xyz[..., 1]>-0.1, xyz[..., 1]<=1.))
    
    # 
    xyz_mask = torch.logical_and(xyz_mask, un == False)
    return xyz_mask

def return_ground(direction):
    # default all False
    # return torch.zeros_like(direction[..., 1]).bool()
    return direction[..., 1] >= 0.
    
    

