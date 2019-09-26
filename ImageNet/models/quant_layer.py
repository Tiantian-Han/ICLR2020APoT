import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# this function construct an additive pot quantization levels set, with clipping threshold = 1,
def build_power_value(B=2):
    value_a = [0.]
    value_b = [0.]
    value_c = [0.]
    for i in range(3):
        if B == 2:
            value_a.append(2 ** (-i - 1))
        elif B == 4:
            value_a.append(2 ** (-2 * i - 1))
            value_b.append(2 ** (-2 * i - 2))
        elif B == 6:
            value_a.append(2 ** (-3 * i - 1))
            value_b.append(2 ** (-3 * i - 2))
            value_c.append(2 ** (-3 * i - 3))
        else:
            pass

    value_s = []
    for a in value_a:
        for b in value_b:
            for c in value_c:
                value_s.append((a + b + c))
    value_s = torch.Tensor(list(set(value_s)))
    levels = value_s.mul(1.0 / torch.max(value_s))
    return levels


def uniform_quantization(b):

    def uniform_quant(x, b=2):
        xdiv = x.mul((2 ** b - 1))
        xhard = xdiv.round().div(2 ** b - 1)
        # xout = (xhard - x).detach() + x
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)
            input_c = input.clamp(min=-1, max=1)
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input.abs() > 1.).float()
            sign = input.sign()
            grad_alpha = (grad_output * (sign * i + (input_q - input) * (1 - i))).sum()
            return grad_input, grad_alpha

    return _uq().apply


def power_quantization(levels):

    def power_quant(x, levels):
        shape = x.shape
        xhard = x.view(-1)
        levels = levels.type_as(x)
        idxs = (xhard.unsqueeze(0) - levels.unsqueeze(1)).abs().min(dim=0)[1]  # project to nearest quantization level
        xhard = levels[idxs].view(shape)
        return xhard

    class _pq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input.div_(alpha)                                      # weights are first divided by alpha
            input_c = input.clamp(min=-1, max=1)                   # then clipped to [-1,1]
            sign = input_c.sign()
            input_abs = input_c.abs()
            input_q = power_quant(input_abs, levels).mul(sign)     # project to Q(alpha, B)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)                           # rescale to the original range
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()             # grad for weights will not be clipped
            input, input_q = ctx.saved_tensors
            i = (input.abs()>1.).float()
            sign = input.sign()
            grad_alpha = (grad_output*(sign*i + (input_q-input)*(1-i))).sum()    # grad for clipping threshold
            return grad_input, grad_alpha

    return _pq().apply


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, power=True):
        super(weight_quantize_fn, self).__init__()
        assert (w_bit ==3 or w_bit == 5) or w_bit == 32, 'incompatible bit-width! '
        self.w_bit = w_bit
        self.power = power                         # use apot or uniform
        self.value_s = build_power_value(self.w_bit - 1)      # quantization levels set
        self.uniform_q = uniform_quantization(b=self.w_bit-1)
        self.power_q = power_quantization(levels=self.value_s)
        self.register_parameter('wgt_alpha', Parameter(torch.tensor(2.0))) # register the parameter clipping threshold

    def forward(self, weight):
        if self.w_bit == 32:
            weight_q = weight
        else:
            mean = weight.data.mean()
            std = weight.data.std()
            weight = weight.add(-mean).div(std)       # weight normalization
            if self.power:
                weight = self.power_q(weight, self.wgt_alpha)  # apot
            else:
                weight = self.uniform_q(weight, self.wgt_alpha) # uniform
            weight_q = weight
        return weight_q


def act_uniform_quantization(b):

    def uniform_quant(x, b=3):
        xdiv = x.mul(2 ** b - 1)
        xhard = xdiv.round().div(2 ** b - 1)
        return xhard

    class _uq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, alpha):
            input=input.div(alpha)
            input_c = input.clamp(max=1)
            input_q = uniform_quant(input_c, b)
            ctx.save_for_backward(input, input_q)
            input_q = input_q.mul(alpha)
            return input_q

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            input, input_q = ctx.saved_tensors
            i = (input > 1.).float()
            grad_alpha = (grad_output * (i + (input_q - input) * (1 - i))).sum()
            grad_input = grad_input*(1-i)     # clip activations
            return grad_input, grad_alpha     # gradients for activations and alpha

    return _uq().apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.weight_quant = weight_quantize_fn(5, True)
        self.act_quant = act_uniform_quantization(5)
        self.act_alpha = torch.nn.Parameter(torch.tensor(8.0))

    def forward(self, x):
        weight_q = self.weight_quant(self.weight)
        x = self.act_quant(x, self.act_alpha)
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def show_params(self):
        alpha = self.weight_quant.wgt_alpha.data.item()
        print('weight alpha: {:2f}, act alpha: {:2f}'.format(alpha, self.act_alpha.data.item()))
