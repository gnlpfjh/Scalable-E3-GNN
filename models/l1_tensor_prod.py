from typing import Tuple
import torch
from torch import Tensor, cat, empty
from torch.nn import Module
from e3nn.o3 import Irreps, Instruction
from math import sqrt

class L1TensorProduct(Module):
    def __init__(self, in1_irreps: Irreps, out_irreps=None,
                 irrep_normalization="component", path_normalization="element",
                 in1_var: list[float]=None, in2_var: list[float]=None, out_var: list[float]=None) -> None:
        super().__init__()
        assert in1_irreps.lmax == 1
        if out_irreps is not None: assert out_irreps.lmax == 1

        self.iri1 = in1_irreps
        self.iri2 = Irreps.spherical_harmonics(1) # first step only sh # in2_irreps if in2_irreps is not None else in1_irreps
        self.iro = out_irreps if out_irreps is not None else in1_irreps

        self.in1_dim = self.iri1.dim # .dim doesnt work with torchscript (in forward)
        self.in2_dim = self.iri2.dim

        # examine l0 and l1 indices / masks
        self.iri1_l0e = torch.zeros(self.in1_dim, dtype=torch.bool)
        self.iri1_l0o = torch.zeros(self.in1_dim, dtype=torch.bool)
        self.iri1_l1e = torch.zeros(self.in1_dim, dtype=torch.bool)
        self.iri1_l1o = torch.zeros(self.in1_dim, dtype=torch.bool)
        i = 0
        for mir in self.iri1:
            if mir.ir.l == 0:
                mask = self.iri1_l0e if mir.ir.p == 1 else self.iri1_l0o
                mask[i:i+mir.mul] = True
            elif mir.ir.l == 1:
                mask = self.iri1_l1e if mir.ir.p == 1 else self.iri1_l1o
                mask[i:i+mir.dim] = True
            i += mir.dim

        # sh only
        self.iri2_l0e = torch.zeros(self.in2_dim, dtype=torch.bool)
        self.iri2_l1o = torch.zeros(self.in2_dim, dtype=torch.bool)
        i = 0
        for mir in self.iri2:
            if mir.ir.l == 0:
                mask = self.iri2_l0e
                mask[i:i+mir.mul] = True
            elif mir.ir.l == 1:
                mask = self.iri2_l1o
                mask[i:i+mir.dim] = True
            i += mir.dim

        self.iro_l0e = torch.zeros(self.iro.dim, dtype=torch.bool)
        self.iro_l0o = torch.zeros(self.iro.dim, dtype=torch.bool)
        self.iro_l1e = torch.zeros(self.iro.dim, dtype=torch.bool)
        self.iro_l1o = torch.zeros(self.iro.dim, dtype=torch.bool)
        i = 0
        for mir in self.iro:
            if mir.ir.l == 0:
                mask = self.iro_l0e if mir.ir.p == 1 else self.iro_l0o
                mask[i:i+mir.mul] = True
            elif mir.ir.l == 1:
                mask = self.iro_l1e if mir.ir.p == 1 else self.iro_l1o
                mask[i:i+mir.dim] = True
            i += mir.dim

        self.num_i1_l0e = self.iri1_l0e.sum().item() # .item for torchscript - no mixed use as shape with number and tensor
        self.num_i1_l0o = self.iri1_l0o.sum().item()
        self.num_i1_l0 = self.num_i1_l0e + self.num_i1_l0o
        self.dim_i1_l1e = self.iri1_l1e.sum().item()
        self.num_i1_l1e = self.dim_i1_l1e//3
        self.dim_i1_l1o = self.iri1_l1o.sum().item()
        self.num_i1_l1o = self.dim_i1_l1o//3
        self.dim_o_l0e = self.iro_l0e.sum().item()
        self.dim_o_l0o = self.iro_l0o.sum().item()
        self.dim_o_l1e = self.iro_l1e.sum().item()
        self.dim_o_l1o = self.iro_l1o.sum().item()

        # init weights (uniform [-1,1])
        # ir2 is only sh -> *1 here in the 2nd dim
        if (self.num_i1_l0e + self.num_i1_l1o) > 0 and (self.dim_o_l0e > 0):
            self.weights_l0e = torch.nn.Parameter(torch.rand((self.num_i1_l0e + self.num_i1_l1o, self.dim_o_l0e)) * 2 - 1)
        if (self.num_i1_l0o + self.num_i1_l1e) > 0 and (self.dim_o_l0o > 0):
            self.weights_l0o = torch.nn.Parameter(torch.rand((self.num_i1_l0o + self.num_i1_l1e, self.dim_o_l0o)) * 2 - 1)
        if (self.num_i1_l0o + self.num_i1_l1e + self.num_i1_l1o) > 0 and (self.dim_o_l1e > 0):
            self.weights_l1e = torch.nn.Parameter(torch.rand((self.num_i1_l0o + self.num_i1_l1e + self.num_i1_l1o, self.dim_o_l1e//3)) * 2 - 1)
        if (self.num_i1_l0e + self.num_i1_l1o + self.num_i1_l1e) > 0 and (self.dim_o_l1o > 0):
            self.weights_l1o = torch.nn.Parameter(torch.rand((self.num_i1_l0e + self.num_i1_l1o + self.num_i1_l1e, self.dim_o_l1o//3)) * 2 - 1)

        # clebsh goardan coefficients from wigner 3j matrices
        self.cg000 = 1
        self.cg110 = 1 / sqrt(3)
        self.cg011 = self.cg110
        self.cg111 = 1 / sqrt(6) # = 1/sqrt(2) * cg110

        # calc norm factors
        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.iri1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(self.iri1), "Len of ir1_var must be equal to len(irreps_in1)"

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.iri2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(self.iri2), "Len of ir2_var must be equal to len(irreps_in2)"

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.iro))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(self.iro), "Len of out_var must be equal to len(irreps_out)"

        self.is_norm = irrep_normalization in ("component", "norm") or path_normalization in ("element", "path")
        if not self.is_norm: return
        self.is_comp_norm = irrep_normalization != "norm" and path_normalization != "path"
        torch._assert(self.is_comp_norm, "Not all norms are implemented yet.")

        # normalization factors
        alpha, x, num_ins = [], [], []
        # add also e3nn instructions for compatibility with the e3nn TensorProduct
        self.instructions: list[Instruction] = []
        for io, mir_out in enumerate(self.iro):
            if irrep_normalization == "component":
                alpha.append(mir_out.ir.dim * out_var[io])
            elif irrep_normalization != "norm":
                alpha.append(1)
            
            if path_normalization in ["element", "none"]: # with none for weight init
                x.append(0.)
            elif path_normalization != "path":
                x.append(1)

            num_ins.append(0)

            for ii2, mir_in2 in enumerate(self.iri2):
                for ii1, mir_in1 in enumerate(self.iri1):
                    if (mir_out.ir.l == 0 and (mir_in2.ir.l == mir_in1.ir.l)) or (mir_out.ir.l == 1 and (mir_in2.ir.l | mir_in1.ir.l)) \
                        and (mir_out.ir.p == mir_in2.ir.p * mir_in1.ir.p):
                        if irrep_normalization == "norm":
                            alpha.append(mir_in1.ir.dim * mir_in2.ir.dim * out_var[io])
                        
                        if path_normalization in ["element", "none"]:
                            x[-1] += in1_var[ii1] * in2_var[ii2] * mir_in1.mul * mir_in2.mul
                        elif path_normalization == "path":
                            x.append(in1_var[ii1] * in2_var[ii2] * mir_in1.mul * mir_in2.mul)

                        num_ins[-1] += 1 
                        
                        self.instructions.append(Instruction(ii1, ii2, io, 'uvw', True, alpha[-1], (mir_in1.mul, mir_in2.mul, mir_out.mul)))
            
        if self.is_comp_norm:
            # regster norm as buffer -> on same device as whole module
            self.register_buffer("norm_l0e", torch.empty(self.dim_o_l0e))
            self.register_buffer("norm_l0o", torch.empty(self.dim_o_l0o))
            self.register_buffer("norm_l1e", torch.empty(self.dim_o_l1e))
            self.register_buffer("norm_l1o", torch.empty(self.dim_o_l1o))
            i0e, i0o, i1e, i1o = 0, 0, 0, 0
            for io, (mir_out, ai, xi) in enumerate(zip(self.iro, alpha, x)):
                if path_normalization == "none":
                    a = sqrt(ai)
                    wi = 1 / sqrt(xi)
                else:
                    a = sqrt((ai / xi) if xi > 0 else ai)
                    wi = 1
                with torch.no_grad():
                    if mir_out.ir.l == 0:
                        if mir_out.ir.p == 1:
                            self.norm_l0e[i0e:i0e+mir_out.mul] = a
                            self.weights_l0e[:,i0e:i0e+mir_out.mul].uniform_(-wi, wi)
                            i0e += mir_out.mul
                        else:
                            self.norm_l0o[i0o:i0o+mir_out.mul] = a
                            self.weights_l0o[:,i0o:i0o+mir_out.mul].uniform_(-wi, wi)
                            i0o += mir_out.mul
                    elif mir_out.ir.l == 1:
                        if mir_out.ir.p == 1:
                            self.norm_l1e[i1e:i1e+mir_out.dim] = a
                            self.weights_l1e[:,i1e:i1e+mir_out.mul].uniform_(-wi, wi)
                            i1e += mir_out.dim
                        else:
                            self.norm_l1o[i1o:i1o+mir_out.dim] = a
                            self.weights_l1o[:,i1o:i1o+mir_out.mul].uniform_(-wi, wi)
                            i1o += mir_out.dim
                
                for i, ins in enumerate(self.instructions):
                    if ins.i_out == io:
                        self.instructions[i] = Instruction(*ins[:-2], path_shape=ins.path_shape, path_weight=a) # cant change values in NamedTuple -> new Tuple


    def forward(self, in1: Tensor, in2: Tensor) -> Tensor:
        # self.iri1.dim doesnt work with torchscript
        torch._assert(in1.shape[-1] == self.in1_dim, f"Incorrect last dimension for in1 = {in1.shape[-1]}, required is {self.in1_dim}")
        torch._assert(in2.shape[-1] == self.in2_dim, f"Incorrect last dimension for in2 = {in2.shape[-1]}, required is {self.in2_dim}")
        # auto broadcast works only with sh in2
        
        out = empty((in1.shape[0], len(self.iro_l0e)), device=in1.device, dtype=in1.dtype)
        
        if self.dim_o_l0e > 0:
            o_l0e = []
            o_l0e.append(in1[:, self.iri1_l0e] * in2[:, self.iri2_l0e])
            # o_l0e.append(in1[:, self.iri1_l0o] * in2[:, self.iri2_l0o]) # doesnt exist (yet) as iri2 is sh
            if self.dim_i1_l1o > 0:
                o_l0e.append(self.cg110 * torch.linalg.vecdot(in1[:, self.iri1_l1o].reshape((-1, self.num_i1_l1o, 3)), in2[:, None, self.iri2_l1o]).reshape(-1, self.num_i1_l1o))

            res = cat(o_l0e, -1) @ self.weights_l0e
            # take dtype from res - workaround for amp (bf16-mixed) with lightning - result of matmul/@ with float32 is bf16
            out[:, self.iro_l0e] = res.to(dtype=out.dtype)
            del res
            if self.is_comp_norm:
                out[:, self.iro_l0e] *= self.norm_l0e

        if self.dim_o_l0o > 0: # more rare case
            o_l0o = []
            o_l0o.append(in1[:, self.iri1_l0o] * in2[:, self.iri2_l0e])
            # o_l0e.append(in1[:, self.iri1_l0e] * in2[:, self.iri2_l0o]) # doesnt exist (yet) as iri2 is sh
            if self.dim_i1_l1e > 0:
                o_l0o.append(self.cg110 * torch.linalg.vecdot(in1[:, self.iri1_l1e].reshape((-1, self.num_i1_l1e, 3)), in2[:, None, self.iri2_l1o]).reshape(-1, self.num_i1_l1e))
            out[:, self.iro_l0o] = (cat(o_l0o, -1) @ self.weights_l0o).to(dtype=out.dtype)
            if self.is_comp_norm:
                out[:, self.iro_l0o] *= self.norm_l0o

        if self.dim_o_l1e > 0: # more rare case
            o_l1e = []
            o_l1e.append(self.cg011 * in1[:, self.iri1_l0o, None] * in2[:, None, self.iri2_l1o])
            if self.dim_i1_l1e > 0:
                o_l1e.append(self.cg011 * in1[:, self.iri1_l1e].reshape(-1, self.num_i1_l1e, 3) * in2[:, None, self.iri2_l0e])
            # i1_l0o * i2_l1e doesnt exist (yet) as iri2 is sh
            if self.dim_i1_l1o > 0:
                o_l1e.append(self.cg111 * torch.linalg.cross(in1[:, self.iri1_l1o].reshape(-1, self.num_i1_l1o, 3),  in2[:, None, self.iri2_l1o]))
            # i1_l1e x i2_l1e doesnt exist (yet) as iri2 is sh
            out[:, self.iro_l1e] = torch.tensordot(cat(o_l1e, -2), self.weights_l1e, ([-2], [0])).transpose(-1,-2).reshape(-1, self.dim_o_l1e) # why view doesnt work sometimes? .squeeze().repeat_interleave(3,-1)

            if self.is_comp_norm:
                out[:, self.iro_l1e] *= self.norm_l1e

        if self.dim_o_l1o > 0: # more rare case
            o_l1o = []
            o_l1o.append(self.cg011 * in1[:, self.iri1_l0e, None] * in2[:, None, self.iri2_l1o])
            if self.dim_i1_l1o > 0:
                o_l1o.append(self.cg011 * in1[:, self.iri1_l1o].reshape(-1, self.num_i1_l1o, 3) * in2[:, None, self.iri2_l0e])
            # i1_l0o * i2_l1e doesnt exist (yet) as iri2 is sh
            if self.dim_i1_l1e > 0:
                o_l1o.append(self.cg111 * torch.linalg.cross(in1[:, self.iri1_l1e].reshape(-1, self.num_i1_l1e, 3),  in2[:, None, self.iri2_l1o]))
            # i1_l1o x i2_l1e doesnt exist (yet) as iri2 is sh
            out[:, self.iro_l1o] = torch.tensordot(cat(o_l1o, -2), self.weights_l1o, ([-2], [0])).transpose(-1,-2).reshape(-1, self.dim_o_l1o) # why view doesnt work sometimes? .squeeze().repeat_interleave(3,-1)
            if self.is_comp_norm:
                out[:, self.iro_l1o] *= self.norm_l1o

        return out.contiguous()
