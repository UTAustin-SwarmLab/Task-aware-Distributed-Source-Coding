import torch

def cov_func(X1, X2):
    # Compute covariance matrix between two matrices
    # X1: n1 x N
    # X2: n2 x N
    # return: scalar
    X1_mean = torch.mean(X1, dim=0)
    X2_mean = torch.mean(X2, dim=0)
    X1_centered = X1 - X1_mean
    X2_centered = X2 - X2_mean
    cov = torch.mm(X1_centered.t(), X2_centered) / (X1_centered.shape[0] - 1)
    ### calculate correlation ###
    std1 = torch.std(X1, dim=0)
    std2 = torch.std(X2, dim=0)
    cov = cov / (std1 * std2)
    
    return torch.abs(cov).mean().item()

### main function ###
if __name__ == "__main__":
    X1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
    X2 = torch.tensor([[1, 2, 9], [6, 5, 4]], dtype=torch.float)
    print(cov_func(X1, X2))
