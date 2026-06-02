import os
from collections import OrderedDict
import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling


import sys
sys.path.append("./Documents/competition/MSDCLIPAdapter")

from clipprad.src.models.submodules.nets.clip.config import CLIPTextConfig, CLIPVisionConfig
from clipprad.src.models.submodules.nets.clip.model import CLIPTextModel, CLIPVisionModel
from clipprad.src.models.submodules.nets.tools.quick_gelu import QuickGELU
from clipprad.src.models.clip import clip
from clipprad import logger, PARALLEL_ADAPTER


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class CLIPTextPrompt(nn.Module):
    def __init__(self, config, classnames=None, ctx_init="This image is", n_ctx=4, pretrained_pth=None):
        super().__init__()
        
        self.embed_dim = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.pretrained_pth = pretrained_pth
        
        if "longclip" in self.pretrained_pth:
            config.max_position_embeddings = 248
            self.max_position_embeddings = 248
            self.positional_embedding_res = nn.Parameter(torch.empty(self.max_position_embeddings, config.hidden_size))
            self.mask1 = torch.zeros([248, 1])
            self.mask1[:20, :] = 1
            self.mask2 = torch.zeros([248, 1])
            self.mask2[20:, :] = 1
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_pth)     
        
        self.model = CLIPTextModel(config)
        
        self.classnames = classnames
        self.ctx_init = ctx_init
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.init_weights()
        
    def init_weights(self):
        if "longclip" in self.pretrained_pth:
            pretrained_dict = torch.load(self.pretrained_pth, map_location="cpu")
            pretrained_text_dict_tmp = {}
            for param_key in pretrained_dict.keys():
                if "visual" not in param_key and "logit_scale" not in param_key and "text_projection" not in param_key:
                    pretrained_text_dict_tmp[param_key] = pretrained_dict[param_key]
            pretrained_text_dict = {}
            for param_key in pretrained_text_dict_tmp.keys():
                if param_key == "positional_embedding_res":
                    pretrained_text_dict[param_key] = pretrained_dict[param_key]
                elif param_key == "positional_embedding":
                    pretrained_text_dict[f"model.text_model.embeddings.position_embedding.weight"] = pretrained_dict[param_key]
                elif param_key == "token_embedding.weight":
                    pretrained_text_dict[f"model.text_model.embeddings.token_embedding.weight"] = pretrained_dict[param_key]
                elif "ln_final" in param_key:
                    pretrained_text_dict[f"{param_key.replace('ln_final', 'model.text_model.final_layer_norm')}"] = pretrained_dict[param_key]
                elif param_key.startswith("transformer.resblocks"):
                    new_param_key = param_key.replace("transformer.resblocks", "model.text_model.encoder.layers")
                    if "in_proj_weight" in param_key:
                        new_param_key = new_param_key.replace("attn", "self_attn")
                        query_weight, key_weight, value_weight = pretrained_dict[param_key].chunk(3, dim=0)
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_weight', 'q_proj.weight')}"]= query_weight
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_weight', 'k_proj.weight')}"]= key_weight
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_weight', 'v_proj.weight')}"]= value_weight
                    if "in_proj_bias" in param_key:
                        new_param_key = new_param_key.replace("attn", "self_attn")
                        query_bias, key_bias, value_bias = pretrained_dict[param_key].chunk(3, dim=0)
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_bias', 'q_proj.bias')}"]= query_bias
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_bias', 'k_proj.bias')}"]= key_bias
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_bias', 'v_proj.bias')}"]= value_bias
                    if "out_proj" in param_key:
                        new_param_key = new_param_key.replace("attn", "self_attn")
                        pretrained_text_dict[new_param_key] = pretrained_dict[param_key]
                    if "ln_1" in param_key:
                        pretrained_text_dict[f"{new_param_key.replace('ln_1', 'layer_norm1')}"] = pretrained_dict[param_key]
                    if "ln_2" in param_key:
                        pretrained_text_dict[f"{new_param_key.replace('ln_2', 'layer_norm2')}"] = pretrained_dict[param_key]
                    if "mlp" in param_key:
                        if "c_fc" in param_key:
                            pretrained_text_dict[f"{new_param_key.replace('c_fc', 'fc1')}"] = pretrained_dict[param_key]
                        if "c_proj" in param_key:
                            pretrained_text_dict[f"{new_param_key.replace('c_proj', 'fc2')}"] = pretrained_dict[param_key]
            del pretrained_dict, pretrained_text_dict_tmp
        else:
            pretrained_dict = torch.load(os.path.join(self.pretrained_pth, "pytorch_model.bin"), map_location="cpu")
            pretrained_text_dict = {}
            for param_key in pretrained_dict.keys():
                if "text_model" in param_key:
                    pretrained_text_dict[f"model.{param_key}"] = pretrained_dict[param_key]
            del pretrained_dict
        
        msg = self.load_state_dict(pretrained_text_dict, strict=False)
        logger.info('Missing keys: {}'.format(msg.missing_keys))
        
        no_adapter_params = []
        for key_name in msg.missing_keys:
            if "adapter" not in key_name:
                no_adapter_params.append(key_name)
        logger.info(f'No adapter parmas: {no_adapter_params}. ')
        
        for n, m in self.named_parameters():
            if n in pretrained_text_dict.keys():
                m.requires_grad = False
                
        logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
        logger.info(f"=> loaded successfully '{self.pretrained_pth}'")
        del pretrained_text_dict
        torch.cuda.empty_cache()
        
        if self.ctx_init is not None:
            self.n_ctx = len(self.ctx_init.split(" "))
            if "longclip" in self.pretrained_pth:
                prompt = clip.tokenize(self.ctx_init, context_length=self.max_position_embeddings)
                with torch.no_grad():
                    embedding = self.model.text_model.embeddings.token_embedding(prompt).type(self.dtype())
            else:
                prompt = self.tokenizer(self.ctx_init, return_tensors="pt")
                with torch.no_grad():
                    embedding = self.model.text_model.embeddings.token_embedding(prompt["input_ids"]).type(self.dtype())
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :]
        else:
            ctx_vectors = torch.empty(self.n_ctx, self.embed_dim, dtype=self.dtype())
            nn.init.normal_(ctx_vectors, std=0.02)
            self.ctx_init = " ".join(["X"] * self.n_ctx)
            
        logger.info(f"Default prompt prefix: {self.ctx_init}")
        logger.info(f"Length of prompt prefix: {self.n_ctx}")
        self.ctx_params = nn.Parameter(ctx_vectors)
        
        prompts = [self.ctx_init + " " + name + "." for name in self.classnames]
        if "longclip" in self.pretrained_pth:
            tokenized_prompts_tmp = torch.cat([clip.tokenize(p, context_length=self.max_position_embeddings) for p in prompts])  # (n_cls, n_tkn)
            with torch.no_grad():
                embedding = self.model.text_model.embeddings.token_embedding(tokenized_prompts_tmp).type(self.dtype())
            tokenized_prompts = {}
            tokenized_prompts["input_ids"] = tokenized_prompts_tmp
            tokenized_prompts["attention_mask"] = None
        else:
            tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", max_length=self.max_position_embeddings, padding="max_length")     # (n_cls, n_token)
            with torch.no_grad():
                embedding = self.model.text_model.embeddings.token_embedding(tokenized_prompts["input_ids"]).type(self.dtype())
            
        self.register_buffer("token_prefix", embedding[:, : 1, :])              # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS
        self.register_buffer("input_ids", tokenized_prompts["input_ids"])
        self.register_buffer("attention_mask", tokenized_prompts["attention_mask"])
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(self.max_position_embeddings).expand((1, -1)))
        
    def dtype(self):
        return self.model.text_model.embeddings.token_embedding.weight.dtype
    
    def encode_token(self, token):
        x = self.model.text_model.embeddings.token_embedding(token)
        return x
        
    def encode_text(self, token_emb):
        bsz, seq_len = self.input_ids.shape
        position_ids = self.position_ids[:, :seq_len]
        
        if "longclip" in self.pretrained_pth:
            x = token_emb.type(self.dtype()) + (self.model.text_model.embeddings.position_embedding(position_ids).to(token_emb.device).type(self.dtype()) * self.mask1.to(token_emb.device)).type(self.dtype()).to(token_emb.device) + (self.positional_embedding_res.to(token_emb.device) * self.mask2.to(token_emb.device)).type(self.dtype()).to(token_emb.device) 
        else:
            x = token_emb.type(self.dtype()) + self.model.text_model.embeddings.position_embedding(position_ids).type(self.dtype()) # （2, 77, 512)
        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self.model.text_model._build_causal_attention_mask(bsz, seq_len, x.dtype).to(x.device)
        if self.attention_mask is not None:
            attention_mask = _expand_mask(self.attention_mask, x.dtype)
        else:
            attention_mask = None
        
        # x = x.permute(1, 0, 2)  # NLD -> LND
        encoder_outputs = self.model.text_model.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.model.text_model.final_layer_norm(last_hidden_state)
        
        x = self.model.text_model.final_layer_norm(x).type(self.dtype())
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            self.input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
        ]
        
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    
    def forward(self):
        ctx = self.ctx_params  # (n_ctx, emb_dim)
        # if ctx.dim() == 2:
        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)   # (n_cls, -1, -1)
        
        prompts = torch.cat(
            [
                self.token_prefix,      # (n_cls, 1, emb_dim)
                ctx,                    # (n_cls, n_ctx, emb_dim)
                self.token_suffix,      # (n_cls, * dim)
            ],
            dim=1,
        )
        text_feat = self.encode_text(prompts)
        return text_feat
    

class LongCLIPTextAdapter(nn.Module):
    def __init__(self, config, pretrained_pth=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.return_tokenizer = config.return_tokenizer
        
        self.pretrained_pth = pretrained_pth
        
        if "longclip" in self.pretrained_pth:
            config.max_position_embeddings = 248
            self.max_position_embeddings = 248
            self.positional_embedding_res = nn.Parameter(torch.empty(self.max_position_embeddings, config.hidden_size))
            self.mask1 = torch.zeros([248, 1])
            self.mask1[:20, :] = 1
            self.mask2 = torch.zeros([248, 1])
            self.mask2[20:, :] = 1
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_pth)     
        
        self.model = CLIPTextModel(config)
        
        self.init_weights()
        
    def init_weights(self):
        if "longclip" in self.pretrained_pth:
            pretrained_dict = torch.load(self.pretrained_pth, map_location="cpu")
            pretrained_text_dict_tmp = {}
            for param_key in pretrained_dict.keys():
                if "visual" not in param_key and "logit_scale" not in param_key and "text_projection" not in param_key:
                    pretrained_text_dict_tmp[param_key] = pretrained_dict[param_key]
            pretrained_text_dict = {}
            for param_key in pretrained_text_dict_tmp.keys():
                if param_key == "positional_embedding_res":
                    pretrained_text_dict[param_key] = pretrained_dict[param_key]
                elif param_key == "positional_embedding":
                    pretrained_text_dict[f"model.text_model.embeddings.position_embedding.weight"] = pretrained_dict[param_key]
                elif param_key == "token_embedding.weight":
                    pretrained_text_dict[f"model.text_model.embeddings.token_embedding.weight"] = pretrained_dict[param_key]
                elif "ln_final" in param_key:
                    pretrained_text_dict[f"{param_key.replace('ln_final', 'model.text_model.final_layer_norm')}"] = pretrained_dict[param_key]
                elif param_key.startswith("transformer.resblocks"):
                    new_param_key = param_key.replace("transformer.resblocks", "model.text_model.encoder.layers")
                    if "in_proj_weight" in param_key:
                        new_param_key = new_param_key.replace("attn", "self_attn")
                        query_weight, key_weight, value_weight = pretrained_dict[param_key].chunk(3, dim=0)
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_weight', 'q_proj.weight')}"]= query_weight
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_weight', 'k_proj.weight')}"]= key_weight
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_weight', 'v_proj.weight')}"]= value_weight
                    if "in_proj_bias" in param_key:
                        new_param_key = new_param_key.replace("attn", "self_attn")
                        query_bias, key_bias, value_bias = pretrained_dict[param_key].chunk(3, dim=0)
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_bias', 'q_proj.bias')}"]= query_bias
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_bias', 'k_proj.bias')}"]= key_bias
                        pretrained_text_dict[f"{new_param_key.replace('in_proj_bias', 'v_proj.bias')}"]= value_bias
                    if "out_proj" in param_key:
                        new_param_key = new_param_key.replace("attn", "self_attn")
                        pretrained_text_dict[new_param_key] = pretrained_dict[param_key]
                    if "ln_1" in param_key:
                        pretrained_text_dict[f"{new_param_key.replace('ln_1', 'layer_norm1')}"] = pretrained_dict[param_key]
                    if "ln_2" in param_key:
                        pretrained_text_dict[f"{new_param_key.replace('ln_2', 'layer_norm2')}"] = pretrained_dict[param_key]
                    if "mlp" in param_key:
                        if "c_fc" in param_key:
                            pretrained_text_dict[f"{new_param_key.replace('c_fc', 'fc1')}"] = pretrained_dict[param_key]
                        if "c_proj" in param_key:
                            pretrained_text_dict[f"{new_param_key.replace('c_proj', 'fc2')}"] = pretrained_dict[param_key]
            del pretrained_dict, pretrained_text_dict_tmp
        else:
            pretrained_dict = torch.load(os.path.join(self.pretrained_pth, "pytorch_model.bin"), map_location="cpu")
            pretrained_text_dict = {}
            for param_key in pretrained_dict.keys():
                if "text_model" in param_key:
                    pretrained_text_dict[f"model.{param_key}"] = pretrained_dict[param_key]
            del pretrained_dict
        
        msg = self.load_state_dict(pretrained_text_dict, strict=False)
        logger.info('Missing keys: {}'.format(msg.missing_keys))
        
        no_adapter_params = []
        for key_name in msg.missing_keys:
            if "adapter" not in key_name:
                no_adapter_params.append(key_name)
        logger.info(f'No adapter parmas: {no_adapter_params}. ')
        
        for n, m in self.named_parameters():
            if n in pretrained_text_dict.keys():
                m.requires_grad = False
                
        logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
        logger.info(f"=> loaded successfully '{self.pretrained_pth}'")
        del pretrained_text_dict
        torch.cuda.empty_cache()
        
    def forward(self, text):
        if "longclip" in self.pretrained_pth:
            input_ids = clip.tokenize(text, context_length=self.max_position_embeddings, truncate=True)
            prompt = {}
            prompt["input_ids"] = input_ids.to(self.model.device)
            prompt["attention_mask"] = (torch.arange(self.max_position_embeddings).expand(input_ids.shape) <= input_ids.argmax(dim=1).unsqueeze(1)).to(self.model.device)
            if self.return_tokenizer:
                return prompt, self.model(prompt["input_ids"], attention_mask=None, return_dict=True)
            else:
                return self.model(prompt["input_ids"], attention_mask=None, return_dict=True)
        else:
            prompt = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            prompt["input_ids"] = prompt["input_ids"].to(self.model.device)
            prompt["attention_mask"] = prompt["attention_mask"].to(self.model.device)
            if self.return_tokenizer:
                return prompt, self.model(prompt["input_ids"], prompt["attention_mask"], return_dict=True)
            else:
                return self.model(prompt["input_ids"], prompt["attention_mask"], return_dict=True)
        

class CLIPImageAdapter(nn.Module):
    def __init__(self, config, pretrained_pth):
        super().__init__()
        self.model = CLIPVisionModel(config)
        self.pretrained_pth = pretrained_pth
        self.init_weights()
        
    def init_weights(self):
        if "longclip" in self.pretrained_pth:
            pretrained_dict = torch.load(self.pretrained_pth, map_location="cpu")
            pretrained_visual_dict_tmp = {}
            for param_key in pretrained_dict.keys():
                if "visual" not in param_key and "logit_scale" not in param_key and "text_projection" not in param_key:
                    pretrained_visual_dict_tmp[param_key] = pretrained_dict[param_key]
            pretrained_visual_dict = {}
        else:
            pretrained_dict = torch.load(os.path.join(self.pretrained_pth, "pytorch_model.bin"), map_location="cpu")
            pretrained_visual_dict = {}
            for param_key in pretrained_dict.keys():
                if "vision_model" in param_key:
                    pretrained_visual_dict[f"model.{param_key}"] = pretrained_dict[param_key]
            del pretrained_dict
            
            msg = self.load_state_dict(pretrained_visual_dict, strict=False)
            logger.info('Missing keys: {}'.format(msg.missing_keys))
            
            adapter_params, no_adapter_params = [], []
            for key_name in msg.missing_keys:
                if "adapter" in key_name:
                    adapter_params.append(key_name)
                else:
                    no_adapter_params.append(key_name)
            logger.info(f'Adapter parmas: {adapter_params}. ')
            logger.info(f'No adapter parmas: {no_adapter_params}. ')
            
            for n, m in self.named_parameters():
                if n in pretrained_visual_dict.keys():
                    m.requires_grad = False
            logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
            logger.info(f"=> loaded successfully '{self.pretrained_pth}'")
            del pretrained_visual_dict
            torch.cuda.empty_cache()
    
    def forward(self, pixel_values: Optional[torch.FloatTensor] = None):
        return self.model(pixel_values, return_dict=True)


class FinedCLIP(nn.Module):
    def __init__(self, text_prompt, ctx_init="This image is", pretrained_pth="pretrained_llms/openai/clip-vit-base-patch32", text_pretrained_pth=None, prompt_n_ctx=8, adapter_mlp_ratio=8, projection_dim=512, fusion_expand=4, fusion_drp=0.5, att_adapter_pos=PARALLEL_ADAPTER, mlp_adapter_pos=PARALLEL_ADAPTER):
        super().__init__()
        self.pretrained_pth = pretrained_pth
        self.text_pretrained_pth = text_pretrained_pth
        self.prompt_n_ctx = prompt_n_ctx
        self.adapter_mlp_ratio = adapter_mlp_ratio
        
        self.projection_dim = projection_dim
        self.fusion_expand = fusion_expand
        self.fusion_drp = fusion_drp
        
        if text_pretrained_pth is not None:
            if "clip" in text_pretrained_pth:
                logger.info("load Text branch...")
                self.text_branch_config = CLIPTextConfig.from_pretrained(pretrained_pth, att_adapter_pos=att_adapter_pos, mlp_adapter_pos=mlp_adapter_pos, adapter_mlp_ratio=adapter_mlp_ratio)
                self.text_branch_model = LongCLIPTextAdapter(self.text_branch_config, pretrained_pth=text_pretrained_pth)
                logger.info("load Caption branch...")
                self.caption_branch_config = CLIPTextConfig.from_pretrained(pretrained_pth, att_adapter_pos=att_adapter_pos, mlp_adapter_pos=mlp_adapter_pos, adapter_mlp_ratio=adapter_mlp_ratio)
                self.caption_branch_model = LongCLIPTextAdapter(self.text_branch_config, pretrained_pth=text_pretrained_pth)
            else:
                logger.info("Error!")
        else:
            self.text_branch_model = nn.Identity()
            self.caption_branch_model = nn.Identity()
            
        logger.info("load Visaul branch...")
        self.visual_branch_config = CLIPVisionConfig.from_pretrained(pretrained_pth, att_adapter_pos=att_adapter_pos, mlp_adapter_pos=mlp_adapter_pos, adapter_mlp_ratio=adapter_mlp_ratio)
        self.visual_branch_model = CLIPImageAdapter(self.visual_branch_config, pretrained_pth)
        
        logger.info("load prompt branch...")
        self.prompt_branch_config = CLIPTextConfig.from_pretrained(pretrained_pth, adapter_pos=None)
        self.prompt_branch_model = CLIPTextPrompt(config=self.prompt_branch_config, classnames=text_prompt, ctx_init=ctx_init, n_ctx=prompt_n_ctx, pretrained_pth=pretrained_pth)
        
        self.visual_projection = nn.Linear(self.visual_branch_config.hidden_size, self.projection_dim, bias=False)
        self.prompt_projection = nn.Linear(self.prompt_branch_config.hidden_size, self.projection_dim, bias=False)
        
        if text_pretrained_pth is not None:
            self.text_projection = nn.Linear(self.text_branch_config.hidden_size, self.projection_dim, bias=False)
            self.caption_projection = nn.Linear(self.caption_branch_config.hidden_size, self.projection_dim, bias=False)
        else:
            self.text_projection = nn.Identity()
            self.caption_projection = nn.Identity()
            
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        
        if text_pretrained_pth is not None:
            self.fusion_adapter_first = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(self.projection_dim * 2, self.projection_dim * self.fusion_expand)),
                ("gelu", QuickGELU()),
                ("drop", nn.Dropout(self.fusion_drp)),
                ("fc2", nn.Linear(self.projection_dim * self.fusion_expand, self.projection_dim))
            ]))
            self.fusion_adapter_second = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(self.projection_dim * 2, self.projection_dim * self.fusion_expand)),
                ("gelu", QuickGELU()),
                ("drop", nn.Dropout(self.fusion_drp)),
                ("fc2", nn.Linear(self.projection_dim * self.fusion_expand, self.projection_dim))
            ]))
            
        else:
            self.fusion_adapter_first = nn.Identity()
            self.fusion_adapter_second = nn.Identity()
        
        self.logger_out()
        
    def logger_out(self):
        # 查看所有的一级子模块的名字和可训练参数量
        for name, module in self.named_children():
            total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            logger.info('^' * 10 + f" Module name: {name}, Trainable parameters: {total_params}, {round(total_params / 1e6, 2)} M" + '^' * 10)
            
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info('^' * 10 + f" Num of learnable params of {self.__class__.__name__}: {n_parameters}, {round(n_parameters / 1e6, 2)} M" + '^' * 10)
        logger.info('=' * 50)
        
        
    def forward(self, pixel_values, batch_info):
        vision_outputs = self.visual_branch_model(pixel_values=pixel_values)
        prompt_outputs = self.prompt_branch_model()
        text_outputs = self.text_branch_model(batch_info["text"])
        caption_outputs = self.caption_branch_model(batch_info["frame_caption"])
        
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        prompt_embeds = prompt_outputs[1]
        prompt_embeds = self.prompt_projection(prompt_embeds)
        
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        prompt_embeds = prompt_embeds / prompt_embeds.norm(p=2, dim=-1, keepdim=True)
        
        if self.text_pretrained_pth is not None:
            text_embeds = text_outputs[1]
            text_embeds = self.text_projection(text_embeds)
            caption_embeds = caption_outputs[1]
            caption_embeds = self.caption_projection(caption_embeds)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            caption_embeds = caption_embeds / caption_embeds.norm(p=2, dim=-1, keepdim=True)
            
            image_embeds = self.fusion_adapter_first(torch.cat([image_embeds, caption_embeds], dim=-1))
            image_embeds = self.fusion_adapter_second(torch.cat([image_embeds, text_embeds], dim=-1))
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(prompt_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        return logits_per_image        
    