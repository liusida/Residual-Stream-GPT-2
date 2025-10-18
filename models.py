import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

@dataclass
class Output:
    pass

class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position embeddings
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: share weights between lm_head and wte
        self.wte.weight = self.lm_head.weight
        self.output = None

    def forward(self, idx, targets=None, keep_output=False, block_mask=None):
        if keep_output:
            self.output = Output()
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        if keep_output:
            self.output.tok_emb = tok_emb
        pos_emb = self.wpe(pos)
        if keep_output:
            self.output.pos_emb = pos_emb
        x = self.drop(tok_emb + pos_emb)
        if keep_output:
            self.output.dropped_emb = x
            self.output.hidden_states = []
        
        # Apply block mask if provided
        if block_mask is None:
            block_mask = [True] * len(self.h)  # All blocks enabled by default
        
        for i, block in enumerate(self.h):
            if block_mask[i]:  # Only process enabled blocks
                x = block(x)
            if keep_output:
                self.output.hidden_states.append(x)
        
        x = self.ln_f(x)
        if keep_output:
            self.output.final_emb = x
        logits = self.lm_head(x)
        if keep_output:
            self.output.logits = logits
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=5, block_mask=None):
        for _ in range(max_new_tokens):
            logits, loss = self(idx, block_mask=block_mask)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_idx, -1, ix)
            idx = torch.cat((idx, xcol), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_name='gpt2'):
        """Load pre-trained GPT-2 model from Hugging Face"""
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        config = GPT2Config()
        model = cls(config)
        
        # Direct parameter mapping
        hf_state_dict = hf_model.state_dict()
        our_state_dict = {}
        
        for name, param in hf_state_dict.items():
            if name.startswith('transformer.'):
                new_name = name.replace('transformer.', '')
                if any(new_name.endswith(w) for w in transposed):
                    our_state_dict[new_name] = param.t()
                else:
                    our_state_dict[new_name] = param
        
        model.load_state_dict(our_state_dict, strict=False)
        return model

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = F.dropout(y, p=self.dropout, training=self.training)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = config.dropout

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x