#%% imports
from e3nn.o3 import FullyConnectedTensorProduct, Irreps
from e3nn.util.jit import script
from torch_geometric import compile
from l1_tensor_prod import L1TensorProduct
import torch
from timeit import timeit

#%% initialize
# iri1 = Irreps("1x0e + 1x1o")
iri1 = Irreps("32x0e + 32x1o")
# iri1 = Irreps("128x0e + 128x1o")
iri2 = Irreps.spherical_harmonics(1)
# iro = Irreps("1x0e + 1x1o")
iro = Irreps("32x0e + 32x1o")
# iro = Irreps("128x0e + 64x1o")
e3nntp = FullyConnectedTensorProduct(iri1, iri2, iro)
l1tp = L1TensorProduct(iri1, iro)
batchsize = 3000

# same wheigts
print("e3nn weight shape", e3nntp.weight.shape)
e3nntp.weight.data = torch.ones_like(e3nntp.weight)
print("l1 weights shape:")
print("l0e", l1tp.weights_l0e.shape)
print("l0o", l1tp.weights_l0o.shape)
print("l1e", l1tp.weights_l1e.shape)
print("l1o", l1tp.weights_l1o.shape)
l1tp.weights_l0e.data = torch.ones_like(l1tp.weights_l0e)
l1tp.weights_l0o.data = torch.ones_like(l1tp.weights_l0o)
l1tp.weights_l1e.data = torch.ones_like(l1tp.weights_l1e)
l1tp.weights_l1o.data = torch.ones_like(l1tp.weights_l1o)

in1 = iri1.randn(batchsize,-1)
# in1 = torch.ones((3, iri1.dim))
in2 = iri2.randn(batchsize,-1)
# in2 = torch.ones((3, iri2.dim))

#%% compare
print("e3nn:\n", e3nntp.cpu()(in1, in2))
print("l1:\n", l1tp.cpu()(in1, in2))

# %% script + cuda
device = "cuda:3" # "cpu" #
dtype = in1.dtype
example_inputs = (
    torch.zeros((4, iri1.dim), dtype=dtype, device=device),
    torch.zeros((4, iri2.dim), dtype=dtype, device=device),
)
check_inputs = (example_inputs, (
    torch.zeros((6, iri1.dim), dtype=dtype, device=device),
    torch.zeros((6, iri2.dim), dtype=dtype, device=device),
))

e3nntp.to(device)
e3t = torch.jit.trace(e3nntp.to(device), example_inputs, check_inputs=check_inputs)
e3s = script(e3nntp).to(device)
e3c = compile(e3nntp, backend="cudagraphs").to(device)

l1tp.to(device)
l1t = torch.jit.trace(l1tp.to(device), example_inputs, check_inputs=check_inputs)
l1s = script(l1tp).to(device)
l1c = compile(l1tp, backend="cudagraphs").to(device)
in1c = in1.to(device)
in2c = in2.to(device)

# %% time
print(timeit(lambda: e3s(in1c, in2c), number=100))
#, 'gc.enable()'
print(timeit(lambda: l1s(in1c, in2c), number=100))

# %%
torch.backends.opt_einsum.enabled
torch.backends.opt_einsum.strategy
# %% equivariance test
from e3nn.util import test as teq
err_eq_l1 = teq.equivariance_error(l1tp, [iri1.randn(2317,-1, device=device), iri2.randn(2317,-1, device=device)], [iri1, iri2], iro, 25, True, True)
print(teq.format_equivariance_error(err_eq_l1))
print("assert l1:", teq.assert_equivariant(l1tp, [iri1.randn(2317,-1, device=device), iri2.randn(2317,-1, device=device)], [iri1, iri2], iro))
err_eq_e3 = teq.equivariance_error(e3nntp, [iri1.randn(2317,-1, device=device), iri2.randn(2317,-1, device=device)], [iri1, iri2], iro, 25, True, True)
print("assert e3:", teq.assert_equivariant(e3nntp, [iri1.randn(2317,-1, device=device), iri2.randn(2317,-1, device=device)], [iri1, iri2], iro))
print(err_eq_e3)
# %% 
