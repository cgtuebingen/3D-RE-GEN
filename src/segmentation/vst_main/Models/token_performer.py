"""
Take Performer as T2T Transformer
"""
import math
import torch
import torch.nn as nn


class crosstask_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        # self.kqv = nn.Linear(dim, 3 * self.emb)
        self.q_s = nn.Linear(dim, self.emb)
        self.k_s = nn.Linear(dim, self.emb)
        self.v_s = nn.Linear(dim, self.emb)

        self.q_c = nn.Linear(dim, self.emb)
        self.k_c = nn.Linear(dim, self.emb)
        self.v_c = nn.Linear(dim, self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj_s = nn.Linear(self.emb, self.emb)
        self.proj_c = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1_s = nn.LayerNorm(dim)
        self.norm1_c = nn.LayerNorm(dim)
        self.norm2_s = nn.LayerNorm(self.emb)
        self.norm2_c = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp_s = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m_s = int(self.emb * kernel_ratio)
        self.w_s = torch.randn(self.m_s, self.emb)
        self.w_s = nn.Parameter(nn.init.orthogonal_(self.w_s) * math.sqrt(self.m_s), requires_grad=False)

        self.m_c = int(self.emb * kernel_ratio)
        self.w_c = torch.randn(self.m_c, self.emb)
        self.w_c = nn.Parameter(nn.init.orthogonal_(self.w_c) * math.sqrt(self.m_c), requires_grad=False)

    def prm_exp_s(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m_s) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w_s)

        return torch.exp(wtx - xd) / math.sqrt(self.m_s)

    def prm_exp_c(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m_c) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w_c)

        return torch.exp(wtx - xd) / math.sqrt(self.m_c)

    def cross_attn(self, saliency_fea, contour_fea):
        k_s, q_s, v_s = self.k_s(saliency_fea), self.q_s(saliency_fea), self.v_s(saliency_fea)
        k_c, q_c, v_c = self.k_c(contour_fea), self.q_c(contour_fea), self.v_c(contour_fea)

        kp_s, qp_s = self.prm_exp_s(k_c), self.prm_exp_s(q_s)  # (B, T, m), (B, T, m)
        D_s = torch.einsum('bti,bi->bt', qp_s, kp_s.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv_s = torch.einsum('bin,bim->bnm', v_c.float(), kp_s)  # (B, emb, m)
        y_s = torch.einsum('bti,bni->btn', qp_s, kptv_s) / (D_s.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # y_s = saliency_fea + self.dp(self.proj_s(y_s))  # same as token_transformer in T2T layer, use v as skip connection
        y_s = self.dp(self.proj_s(y_s))  # same as token_transformer in T2T layer, use v as skip connection

        kp_c, qp_c = self.prm_exp_c(k_s), self.prm_exp_c(q_c)  # (B, T, m), (B, T, m)
        D_c = torch.einsum('bti,bi->bt', qp_c, kp_c.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv_c = torch.einsum('bin,bim->bnm', v_s.float(), kp_c)  # (B, emb, m)
        y_c = torch.einsum('bti,bni->btn', qp_c, kptv_c) / (D_c.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # y_c = contour_fea + self.dp(self.proj_c(y_c))  # same as token_transformer in T2T layer, use v as skip connection
        y_c = self.dp(self.proj_c(y_c))  # same as token_transformer in T2T layer, use v as skip connection

        return y_s, y_c

    def forward(self, saliency_fea, contour_fea):
        # cross task attention
        saliency_fea_fuse, contour_fea_fuse = self.cross_attn(self.norm1_s(saliency_fea), self.norm1_c(contour_fea))

        saliency_fea = saliency_fea + saliency_fea_fuse
        contour_fea = contour_fea + contour_fea_fuse

        saliency_fea = saliency_fea + self.mlp_s(self.norm2_s(saliency_fea))
        contour_fea = contour_fea + self.mlp_c(self.norm2_c(contour_fea))

        return saliency_fea, contour_fea


class Token_performer(nn.Module):
    def __init__(self, dim, in_dim, head_cnt=1, kernel_ratio=0.5, dp1=0.1, dp2 = 0.1):
        super().__init__()
        self.emb = in_dim * head_cnt # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = head_cnt
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(self.emb)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            nn.GELU(),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        # y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection
        y = self.dp(self.proj(y))
        return y

    def forward(self, x):
        x = x + self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

