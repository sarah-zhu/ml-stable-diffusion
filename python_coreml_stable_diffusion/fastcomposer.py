import torch
import torch.nn as nn
import numpy as np
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FastComposerPostfuse(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, text_embeds, object_embeds):
        text_object_embeds = torch.cat([text_embeds, object_embeds], dim=-1)
        text_object_embeds = self.mlp1(text_object_embeds) + text_embeds
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        image_token_mask,
        num_objects,
    ) -> torch.Tensor:
        text_object_embeds = fuse_object_embeddings(
            text_embeds, image_token_mask, object_embeds, num_objects, self.fuse_fn
        )

        return text_object_embeds




def tokenize_and_mask_noun_phrases_ends(tokenizer, caption, image_token):

    input_ids = tokenizer.encode(caption)

    tokenizer.add_tokens([image_token], special_tokens=True)
    image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    noun_phrase_end_mask = [False for _ in input_ids]
    clean_input_ids = []
    clean_index = 0

    for i, id in enumerate(input_ids):
        if id == image_token_id:
            noun_phrase_end_mask[clean_index - 1] = True
        else:
            clean_input_ids.append(id)
            clean_index += 1

    max_len = tokenizer.model_max_length

    if len(clean_input_ids) > max_len:
        clean_input_ids = clean_input_ids[:max_len]
    else:
        clean_input_ids = clean_input_ids + [tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
        )

    if len(noun_phrase_end_mask) > max_len:
        noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
    else:
        noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
        )
    return np.expand_dims(np.array(clean_input_ids), axis=0), np.expand_dims(np.array(noun_phrase_end_mask), axis=0)


