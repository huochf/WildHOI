import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class ActNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))


    def forward(self, x, scale_drift=None, bias_drift=None):
        # x, scale_drift, bias_drift: [b, c] or [b, n, c], 
        if len(x.shape) == 3:
            b, n, c = x.shape
            scale = self.scale.view(1, 1, -1).repeat(b, n, 1)
            bias = self.bias.view(1, 1, -1).repeat(b, n, 1)
        else:
            b, c = x.shape
            scale = self.scale.view(1, -1).repeat(b, 1)
            bias = self.bias.view(1, -1).repeat(b, 1)
        if scale_drift is not None:
            scale = scale + scale_drift
        if bias_drift is not None:
            bias = bias + bias_drift

        z = (x - bias) * torch.exp(-scale)
        logdet = - scale.sum(-1)

        return z, logdet


    def inverse(self, z, scale_drift=None, bias_drift=None):
        if len(z.shape) == 3:
            b, n, c = z.shape
            scale = self.scale.view(1, 1, -1).repeat(b, n, 1)
            bias = self.bias.view(1, 1, -1).repeat(b, n, 1)
        else:
            b, c = z.shape
            scale = self.scale.view(1, -1).repeat(b, 1)
            bias = self.bias.view(1, -1).repeat(b, 1)
        if scale_drift is not None:
            scale = scale + scale_drift
        if bias_drift is not None:
            bias = bias + bias_drift

        x = z * torch.exp(scale) + bias
        logdet = scale.sum(-1)

        return x, logdet


class InvLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        w = torch.randn(dim, dim)
        w = torch.qr(w)[0]
        self.weights = nn.Parameter(w.float())


    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 2:
            x = x.unsqueeze(1)
        weights = self.weights.view(1, self.dim, self.dim).repeat(x_shape[0], 1, 1)
        z = x @ weights
        z = z.view(*x_shape)

        logdet = torch.slogdet(weights)[-1]
        if len(x_shape) == 3:
            logdet = logdet * x_shape[1]

        return z, logdet


    def inverse(self, z):
        z_shape = z.shape
        if len(z_shape) == 2:
            z = z.unsqueeze(1)
        weights = self.weights.inverse()
        weights = weights.view(1, self.dim, self.dim).repeat(z_shape[0], 1, 1)
        x = z @ weights
        x = x.view(*z_shape)

        logdet = torch.slogdet(weights)[-1]
        if len(z_shape) == 3:
            logdet = logdet * z_shape[1]

        return x, logdet


class LULinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.lower_indices = np.tril_indices(dim, k=-1)
        self.upper_indices = np.triu_indices(dim, k=1)
        self.diag_indices = np.diag_indices(dim)

        n_triangular_entries = ((dim -1) * dim) // 2
        self.lower_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.upper_entries = nn.Parameter(torch.zeros(n_triangular_entries))
        self.eps = 1e-3
        constant = np.log(np.exp(1 - self.eps) - 1)
        self.unconstrained_upper_diag = nn.Parameter(torch.ones(dim) * constant)
        self.bias = nn.Parameter(torch.zeros(dim))


    def get_weights(self, inverse=False):
        lower = self.lower_entries.new_zeros(self.dim, self.dim)
        lower[self.lower_indices[0], self.lower_indices[1]] = self.lower_entries
        lower[self.diag_indices[0], self.diag_indices[1]] = 1.0

        upper = self.upper_entries.new_zeros(self.dim, self.dim)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries
        upper_diag = F.softplus(self.unconstrained_upper_diag) + self.eps
        upper[self.diag_indices[0], self.diag_indices[1]] = upper_diag

        if not inverse:
            weights = lower @ upper
        else:
            identity = torch.eye(self.dim, self.dim, dtype=lower.dtype).to(lower.device)
            lower_inverse, _ = torch.triangular_solve(identity, lower, upper=False, unitriangular=True)
            weights, _ = torch.triangular_solve(lower_inverse, upper, upper=True, unitriangular=False)

        return weights


    def forward(self, x):
        # x: [B, C] or [B, N, C]
        weights = self.get_weights(inverse=False)
        z = F.linear(x, weights, self.bias)
        upper_diag = F.softplus(self.unconstrained_upper_diag) + self.eps

        if len(x.shape) == 3:
            logdet = torch.sum(torch.log(upper_diag)) * z.new_ones(z.shape[0], z.shape[1])
        else:
            logdet = torch.sum(torch.log(upper_diag)) * z.new_ones(z.shape[0])

        return z, logdet


    def inverse(self, z):
        weights = self.get_weights(inverse=True)
        x = F.linear(z - self.bias, weights)
        upper_diag = F.softplus(self.unconstrained_upper_diag) + self.eps
        logdet = - torch.sum(torch.log(upper_diag)) * z.new_ones(z.shape[0])

        if len(x.shape) == 3:
            logdet = torch.sum(torch.log(upper_diag)) * z.new_ones(z.shape[0], z.shape[1])
        else:
            logdet = torch.sum(torch.log(upper_diag)) * z.new_ones(z.shape[0])

        return x, logdet


class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout_probability=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim, affine=True)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim, affine=True)
        self.activation = F.relu
        self.dropout = nn.Dropout(p=dropout_probability)
        nn.init.uniform_(self.fc2.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.fc2.bias, -1e-3, 1e-3)


    def forward(self, x):
        # x: [B, C] or [B, N, C]
        x_shape_org = x.shape
        if len(x_shape_org) == 3:
            x = x.flatten(0, 1)
        residual = self.fc1(self.activation(self.bn1(x)))
        residual = self.activation(self.bn2(residual))
        residual = self.fc2(self.dropout(residual))
        
        x = x + residual
        if len(x_shape_org) == 3:
            x = x.reshape(x_shape_org)

        return x


class ResidualNet(nn.Module):

    def __init__(self, dim, hidden_dim, out_dim, num_blocks=2, c_dim=None, dropout_probability=0.0):
        super().__init__()

        if c_dim is not None:
            self.initial_layer = nn.Linear(dim + c_dim, hidden_dim)
        else:
            self.initial_layer = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, dropout_probability) for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_dim, out_dim)
        stdv = 0.01 / np.sqrt(hidden_dim)
        nn.init.uniform_(self.final_layer.weight, -stdv, stdv)
        nn.init.uniform_(self.final_layer.bias, -stdv, stdv)


    def forward(self, x, context=None):
        # x: [B, C] or [B, N, C]
        if context is None:
            x = self.initial_layer(x, context)
        else:
            x = self.initial_layer(torch.cat([x, context], dim=-1))

        for block in self.blocks:
            x = block(x)

        x = self.final_layer(x)
        return x


class MLP(nn.Module):

    def __init__(self, dim, width, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim

        self.linear1 = nn.Linear(dim, width)
        self.actnorm1 = ActNorm(width)
        self.linear2 = nn.Linear(width, width)
        self.actnorm2 = ActNorm(width)
        self.linear3 = nn.Linear(width, dim_out)


    def forward(self, x):
        h = F.relu(self.actnorm1(self.linear1(x))[0])
        h = F.relu(self.actnorm2(self.linear2(h))[0])
        h = self.linear3(h)

        return h


class AffineCoupling(nn.Module):

    def __init__(self, dim, width):
        super().__init__()
        self.mlp = MLP(dim // 2, width, dim)

        self.log_scale_factor = nn.Parameter(torch.zeros(dim))


    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=-1)

        h = self.mlp(x_b)
        if len(h.shape) == 3:
            h = h * self.log_scale_factor.exp().view(1, 1, -1)
        else:
            h = h * self.log_scale_factor.exp().view(1, -1)

        t = h[..., 0::2]
        s = h[..., 1::2]
        s = torch.sigmoid(s) + 0.5

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=-1)

        logdet = s.log().flatten(1).sum(1)

        return z, logdet


    def inverse(self, z):
        z_a, z_b = z.chunk(2, dim=-1)

        h = self.mlp(z_b)
        if len(h.shape) == 3:
            h = h * self.log_scale_factor.exp().view(1, 1, -1)
        else:
            h = h * self.log_scale_factor.exp().view(1, -1)

        t = h[..., 0::2]
        s = h[..., 1::2]
        s = torch.sigmoid(s) + 0.5

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=-1)

        logdet = - s.log().flatten(1).sum(1)
        return x, logdet


class AdditiveCoupling(nn.Module):

    def __init__(self, mask, dim, hidden_dim, num_blocks=2, c_dim=None, dropout_probability=0.0):
        super().__init__()
        self.dim = len(mask)

        features_vector = torch.arange(self.dim)
        self.register_buffer('identity_features', features_vector.masked_select(mask <= 0))
        self.register_buffer('transform_features', features_vector.masked_select(mask > 0))

        self.num_identity_features = len(self.identity_features)
        self.num_transform_features = len(self.transform_features)
        self.transform = ResidualNet(dim=self.num_identity_features, 
                                     hidden_dim=hidden_dim,
                                     out_dim=self.num_transform_features,
                                     num_blocks=num_blocks,
                                     c_dim=c_dim,
                                     dropout_probability=dropout_probability)


    def forward(self, x, context=None):
        # x: [B, C], or [B, N, C]
        z = torch.empty_like(x)

        x_identity = x[..., self.identity_features]
        x_transform = x[..., self.transform_features]
        transform_params = self.transform(x_identity, context)
        bias = transform_params
        scale = torch.ones_like(bias)
        x_transform = scale * x_transform + bias

        z[..., self.identity_features] = x_identity
        z[..., self.transform_features] = x_transform

        logdet = scale.log().sum(-1)

        return z, logdet


    def inverse(self, z, context=None):
        x = torch.empty_like(z)

        z_identity = z[..., self.identity_features]
        z_transform = z[..., self.transform_features]
        transform_params = self.transform(z_identity, context)
        bias = transform_params
        scale = torch.ones_like(bias)
        z_transform = (z_transform - bias) / scale

        x[..., self.identity_features] = z_identity
        x[..., self.transform_features] = z_transform

        logdet = - scale.log().sum(-1)

        return x, logdet


class Gaussianize(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Linear(dim, 2 * dim)
        self.log_scale_factor = nn.Parameter(torch.zeros(2 * dim))
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()


    def forward(self, x1, x2):
        h = self.net(x1) 
        if len(h.shape) == 3:
            h = h * self.log_scale_factor.exp().view(1, 1, -1)
        else:
            h = h * self.log_scale_factor.exp().view(1, -1)
        m, logs = h[..., 0::2], h[..., 1::2]
        z2 = (x2 - m) * torch.exp(-logs)
        logdet = - logs.sum(-1)

        return z2, logdet


    def inverse(self, x1, z2):
        h = self.net(x1)
        if len(h.shape) == 3:
            h = h * self.log_scale_factor.exp().view(1, 1, -1)
        else:
            h = h * self.log_scale_factor.exp().view(1, -1)
        m, logs = h[..., 0::2], h[..., 1::2]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum(-1)

        return x2, logdet


class Attention(nn.Module):

    def __init__(self, qkv_dims, out_dim, head_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        all_head_dim = num_heads * head_dim
        self.scale = all_head_dim ** -0.5

        self.q_embed = nn.Linear(qkv_dims[0], all_head_dim)
        self.k_embed = nn.Linear(qkv_dims[1], all_head_dim)
        self.v_embed = nn.Linear(qkv_dims[2], all_head_dim)

        self.projection = nn.Linear(all_head_dim, out_dim)


    def forward(self, q, k, v, attn_mask):
        # attn_mask: [n_q, n_k]
        b, n_q, _ = q.shape
        b, n_k, _ = k.shape

        q = self.q_embed(q).view(b, n_q, -1, self.num_heads).permute(0, 3, 1, 2) # [b, num_heads, n, c]
        k = self.k_embed(k).view(b, n_k, -1, self.num_heads).permute(0, 3, 1, 2)
        v = self.v_embed(v).view(b, n_k, -1, self.num_heads).permute(0, 3, 1, 2)

        q = q * self.scale
        attn = q @ k.transpose(-1, -2) # [b, num_heads, n_q, n_k]
        attn = attn - 1e8 * (1 - attn_mask.view(1, 1, n_q, n_k))
        attn = attn.softmax(dim=-1)
        v = attn @ v # [b, num_heads, n_q, c]
        v = v.permute(0, 2, 3, 1) # [b, n_q, c, num_heads]
        v = v.reshape(b, n_q, -1)
        v = self.projection(v)

        return v


class AdditiveCouplingSelfAttention(nn.Module):

    def __init__(self, mask, dim, pos_dim, width, c_dim=None, num_blocks=2, head_dim=3, num_heads=8, dropout_probability=0.):
        super().__init__()

        seq_indices = torch.arange(mask.shape[0])
        self.register_buffer('identity_sequences', seq_indices.masked_select(mask <= 0))
        self.register_buffer('transform_sequences', seq_indices.masked_select(mask > 0))

        self.attn1 = Attention([pos_dim, dim, dim], dim, head_dim, num_heads)
        self.attn2 = Attention([pos_dim, dim, dim], dim, head_dim, num_heads)
        self.transform = ResidualNet(dim=dim, 
                                     hidden_dim=width,
                                     out_dim=dim * 2,
                                     num_blocks=num_blocks,
                                     c_dim=c_dim,
                                     dropout_probability=dropout_probability)


    def forward(self, x, pos, attn_mask, context=None):
        # x: [B, N, C], attn_mask: [N, N]
        z = torch.empty_like(x)
        logdet = 0

        x_identity = x[:, self.identity_sequences]
        x_transform = x[:, self.transform_sequences]

        pos_identity = pos[:, self.identity_sequences]
        pos_transform = pos[:, self.transform_sequences]

        h = self.attn1(pos_transform, x_identity, x_identity, attn_mask[self.transform_sequences][:, self.identity_sequences])
        if context is not None:
            h = self.transform(h, context[:, self.transform_sequences])
        else:
            h = self.transform(h)
        scale = h[..., 1::2]
        scale = torch.sigmoid(scale) + 0.5
        bias = h[..., 0::2]
        x_transform = scale * x_transform + bias
        logdet = logdet + scale.log().flatten(1).sum(-1).unsqueeze(1) / z.shape[1]

        h = self.attn2(pos_identity, x_transform, x_transform, attn_mask[self.identity_sequences][:, self.transform_sequences])
        if context is not None:
            h = self.transform(h, context[:, self.identity_sequences])
        else:
            h = self.transform(h)
        scale = h[..., 1::2]
        scale = torch.sigmoid(scale) + 0.5
        bias = h[..., 0::2]
        x_identity = scale * x_identity + bias
        logdet = logdet + scale.log().flatten(1).sum(-1).unsqueeze(1) / x.shape[1]

        z[:, self.identity_sequences] = x_identity
        z[:, self.transform_sequences] = x_transform
        return z, logdet


    def inverse(self, z, pos, attn_mask, context=None):
        x = torch.empty_like(z)
        logdet = 0

        z_identity = z[:, self.identity_sequences]
        z_transform = z[:, self.transform_sequences]
        pos_identity = pos[:, self.identity_sequences]
        pos_transform = pos[:, self.transform_sequences]

        h = self.attn2(pos_identity, z_transform, z_transform, attn_mask[self.identity_sequences][:, self.transform_sequences])
        if context is not None:
            h = self.transform(h, context[:, self.identity_sequences])
        else:
            h = self.transform(h)
        scale = h[..., 1::2]
        scale = torch.sigmoid(scale) + 0.5
        bias = h[..., 0::2]
        z_identity = (z_identity - bias) / scale
        logdet = logdet - scale.log().flatten(1).sum(-1).unsqueeze(1) / z.shape[1]

        h = self.attn1(pos_transform, z_identity, z_identity, attn_mask[self.transform_sequences][:, self.identity_sequences])
        if context is not None:
            h = self.transform(h, context[:, self.transform_sequences])
        else:
            h = self.transform(h)
        scale = h[..., 1::2]
        scale = torch.sigmoid(scale) + 0.5
        bias = h[..., 0::2]
        z_transform = (z_transform - bias) / scale
        logdet = logdet - scale.log().flatten(1).sum(-1).unsqueeze(1) / z.shape[1]

        x[:, self.identity_sequences] = z_identity
        x[:, self.transform_sequences] = z_transform

        return x, logdet
