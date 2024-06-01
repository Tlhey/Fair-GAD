import torch
import torch.nn.functional as F

def sample_dimensions(Z, d):
    idx = torch.randperm(Z.size(1))[:d]
    return Z[:, idx]

p = 0.4
dz = 50
d = 25
a = 0.01
ws = 0.5
num_nodes = 8

S = torch.bernoulli(torch.full((num_nodes,), p))
Z = torch.randn((num_nodes, dz))
v = torch.randn((d,))
X = sample_dimensions(Z, d) + S.unsqueeze(1) * v
cosine_sim = torch.mm(Z, Z.t()) / (torch.norm(Z, dim=1).unsqueeze(1) * torch.norm(Z, dim=1))
P_A = torch.sigmoid(cosine_sim + a * (S.unsqueeze(1) == S).float())
A = torch.bernoulli(P_A)
w = torch.randn((dz,))
Y = torch.mm(Z, w.unsqueeze(1)).squeeze() + ws * torch.mm(A, S.unsqueeze(1)).squeeze() / (A.sum(dim=1) + 1e-6)
Y_mean = Y.mean()
Y = (Y > Y_mean).float()


print("Sensitive attributes S:", S)
print("Latent embeddings Z:", Z)
print("Observed features X:", X)
print("Edge probabilities P_A:", P_A)
print("Adjacency matrix A:", A)
print("Continuous labels Y:", Y)
print("Binary labels Y:", Y)
