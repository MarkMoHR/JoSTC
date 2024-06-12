from typing import Optional, Tuple

import torch
from torch import nn


class HyperNorm(nn.Module):
    def __init__(self, input_size: int, num_units: int, hyper_embedding_size: int, use_bias: bool = True):
        super().__init__()
        self.num_units = num_units
        self.embedding_size = hyper_embedding_size
        self.use_bias = use_bias

        self.z_w = nn.Linear(input_size, self.embedding_size, bias=True)
        self.alpha = nn.Linear(self.embedding_size, self.num_units, bias=False)

        if self.use_bias:
            self.z_b = nn.Linear(input_size, self.embedding_size, bias=False)
            self.beta = nn.Linear(self.embedding_size, self.num_units, bias=False)

    def __call__(self, input: torch.Tensor, hyper_output: torch.Tensor):
        zw = self.z_w(hyper_output)
        alpha = self.alpha(zw)
        result = torch.mul(alpha, input)

        if self.use_bias:
            zb = self.z_b(hyper_output)
            beta = self.beta(zb)
            result = torch.add(result, beta)

        return result


class LSTMCell(nn.Module):
    """
    ## Long Short-Term Memory Cell
    LSTM Cell computes $c$, and $h$. $c$ is like the long-term memory,
    and $h$ is like the short term memory.
    We use the input $x$ and $h$ to update the long term memory.
    In the update, some features of $c$ are cleared with a forget gate $f$,
    and some features $i$ are added through a gate $g$.
    The new short term memory is the $\tanh$ of the long-term memory
    multiplied by the output gate $o$.
    Note that the cell doesn't look at long term memory $c$ when doing the update. It only modifies it.
    Also $c$ never goes through a linear transformation.
    This is what solves vanishing and exploding gradients.
    Here's the update rule.
    \begin{align}
    c_t &= \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t) \\
    h_t &= \sigma(o_t) \odot \tanh(c_t)
    \end{align}
    $\odot$ stands for element-wise multiplication.
    Intermediate values and gates are computed as linear transformations of the hidden
    state and input.
    \begin{align}
    i_t &= lin_x^i(x_t) + lin_h^i(h_{t-1}) \\
    f_t &= lin_x^f(x_t) + lin_h^f(h_{t-1}) \\
    g_t &= lin_x^g(x_t) + lin_h^g(h_{t-1}) \\
    o_t &= lin_x^o(x_t) + lin_h^o(h_{t-1})
    \end{align}
    """

    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False,
                 forget_bias: float = 1.0):
        super().__init__()

        # These are the linear layer to transform the `input` and `hidden` vectors.
        # One of them doesn't need a bias since we add the transformations.
        self.forget_bias = forget_bias

        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        if layer_norm:
            self.layer_norm_all = nn.LayerNorm(4 * hidden_size)
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm_all = nn.Identity()
            self.layer_norm_c = nn.Identity()

    def __call__(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        ifgo = self.input_lin(x) + self.hidden_lin(h)
        ifgo = self.layer_norm_all(ifgo)

        ifgo = ifgo.chunk(4, dim=-1)
        i, j, f, o = ifgo

        g = torch.tanh(j)

        c_next = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * g
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next


class LSTM(nn.Module):
    """
    ## Multilayer LSTM
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        """
        Create a network of `n_layers` of LSTM.
        """

        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size)] +
                                   [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def __call__(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        `x` has shape `[n_steps, batch_size, input_size]` and
        `state` is a tuple of $h$ and $c$, each with a shape of `[batch_size, hidden_size]`.
        """
        n_steps, batch_size = x.shape[:2]

        # Initialize the state if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            # Reverse stack the tensors to get the states of each layer <br />
            # ?? You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))

        # Array to collect the outputs of the final layer at each time step.
        out = []
        for t in range(n_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the layer
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                # Input to the next layer is the state of this layer
                inp = h[layer]
            # Collect the output $h$ of the final layer
            out.append(h[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)


class HyperLSTMCell(nn.Module):
    """
    ## HyperLSTM Cell
    For HyperLSTM the smaller network and the larger network both have the LSTM structure.
    This is defined in Appendix A.2.2 in the paper.
    """

    def __init__(self, input_size: int, num_units: int, hyper_num_units: int = 256, hyper_embedding_size: int = 32,
                 forget_bias: float = 1.0):
        """
        `input_size` is the size of the input $x_t$,
        `num_units` is the size of the LSTM, and
        `hyper_num_units` is the size of the smaller LSTM that alters the weights of the larger outer LSTM.
        `hyper_embedding_size` is the size of the feature vectors used to alter the LSTM weights.
        We use the output of the smaller LSTM to compute $z_h^{i,f,g,o}$, $z_x^{i,f,g,o}$ and
        $z_b^{i,f,g,o}$ using linear transformations.
        We calculate $d_h^{i,f,g,o}(z_h^{i,f,g,o})$, $d_x^{i,f,g,o}(z_x^{i,f,g,o})$, and
        $d_b^{i,f,g,o}(z_b^{i,f,g,o})$ from these, using linear transformations again.
        These are then used to scale the rows of weight and bias tensors of the main LSTM.
        ?? Since the computation of $z$ and $d$ are two sequential linear transformations
        these can be combined into a single linear transformation.
        However we've implemented this separately so that it matches with the description
        in the paper.
        """
        super().__init__()
        self.forget_bias = forget_bias

        self.hyper = LSTMCell(num_units + input_size, hyper_num_units, layer_norm=True)

        self.w_x = nn.Linear(input_size, 4 * num_units, bias=False)
        self.w_h = nn.Linear(num_units, 4 * num_units, bias=False)

        self.hyper_ix = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=False)
        self.hyper_jx = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=False)
        self.hyper_fx = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=False)
        self.hyper_ox = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=False)

        self.hyper_ih = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=True)
        self.hyper_jh = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=True)
        self.hyper_fh = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=True)
        self.hyper_oh = HyperNorm(num_units, num_units, hyper_embedding_size, use_bias=True)

        # zero initialization
        self.bias = nn.Parameter(torch.zeros(4 * num_units))

        self.layer_norm_all = nn.LayerNorm(num_units * 4)
        self.layer_norm_c = nn.LayerNorm(num_units)

    def __call__(self, x: torch.Tensor,
                 h: torch.Tensor, c: torch.Tensor,
                 h_hat: torch.Tensor, c_hat: torch.Tensor):
        hyper_input = torch.cat([x, h], dim=-1)
        h_hat, c_hat = self.hyper(hyper_input, h_hat, c_hat)
        hyper_output = h_hat

        xh = self.w_x(x)
        hh = self.w_h(h)

        ix, jx, fx, ox = torch.chunk(xh, 4, 1)
        ix = self.hyper_ix(ix, hyper_output)
        jx = self.hyper_jx(jx, hyper_output)
        fx = self.hyper_fx(fx, hyper_output)
        ox = self.hyper_ox(ox, hyper_output)

        ih, jh, fh, oh = torch.chunk(hh, 4, 1)
        ih = self.hyper_ih(ih, hyper_output)
        jh = self.hyper_jh(jh, hyper_output)
        fh = self.hyper_fh(fh, hyper_output)
        oh = self.hyper_oh(oh, hyper_output)

        ib, jb, fb, ob = torch.chunk(self.bias, 4, 0)

        i = ix + ih + ib
        j = jx + jh + jb
        f = fx + fh + fb
        o = ox + oh + ob

        concat = torch.cat([i, j, f, o], 1)
        concat = self.layer_norm_all(concat)
        i, j, f, o = torch.chunk(concat, 4, 1)

        g = torch.tanh(j)

        c_next = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * g
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next, h_hat, c_hat


class HyperLSTM(nn.Module):
    """
    # HyperLSTM module
    """
    def __init__(self, input_size: int, hidden_size: int, hyper_size: int, n_z: int, n_layers: int):
        """
        Create a network of `n_layers` of HyperLSTM.
        """

        super().__init__()

        # Store sizes to initialize state
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size

        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([HyperLSTMCell(input_size, hidden_size, hyper_size, n_z)] +
                                   [HyperLSTMCell(hidden_size, hidden_size, hyper_size, n_z) for _ in
                                    range(n_layers - 1)])

    def __call__(self, x: torch.Tensor,
                 state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """
        * `x` has shape `[n_steps, batch_size, input_size]` and
        * `state` is a tuple of $h, c, \hat{h}, \hat{c}$.
         $h, c$ have shape `[batch_size, hidden_size]` and
         $\hat{h}, \hat{c}$ have shape `[batch_size, hyper_size]`.
        """
        n_steps, batch_size = x.shape[:2]

        # Initialize the state with zeros if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            h_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
            c_hat = [x.new_zeros(batch_size, self.hyper_size) for _ in range(self.n_layers)]
        #
        else:
            (h, c, h_hat, c_hat) = state
            # Reverse stack the tensors to get the states of each layer
            #
            # ?? You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            h_hat, c_hat = list(torch.unbind(h_hat)), list(torch.unbind(c_hat))

        # Collect the outputs of the final layer at each step
        out = []
        for t in range(n_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the layer
                h[layer], c[layer], h_hat[layer], c_hat[layer] = \
                    self.cells[layer](inp, h[layer], c[layer], h_hat[layer], c_hat[layer])
                # Input to the next layer is the state of this layer
                inp = h[layer]
            # Collect the output $h$ of the final layer
            out.append(h[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        h_hat = torch.stack(h_hat)
        c_hat = torch.stack(c_hat)

        #
        return out, (h, c, h_hat, c_hat)
