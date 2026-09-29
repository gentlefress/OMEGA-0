"""Load original WAM checkpoints or training exports as (model, metadata)."""
import json
import logging
from pathlib import Path


def load_checkpoint(*, artifact=None, checkpoint=None, external_models=None,
                    device="cuda", attn_implementation=None):
    if (artifact is None) == (checkpoint is None):
        raise ValueError("Specify exactly one of artifact or checkpoint")
    if checkpoint is not None:
        if external_models is None:
            raise ValueError("external_models is required for an original checkpoint")
        return _load_original(checkpoint, external_models, device,
                              attn_implementation or "flash_attention_2")
    if external_models is not None or attn_implementation is not None:
        raise ValueError("external_models and attn_implementation only apply to original checkpoints")
    from omega.models.artifact import load_model
    return load_model(artifact, device)


def _load_original(checkpoint, external_models, device='cuda',
                   attn_implementation='flash_attention_2'):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from omega.models.config import ModelConfig, checkpoint_config
    from omega.models.wam import ACVJEPAModel
    checkpoint, external_models = Path(checkpoint).expanduser(), Path(external_models).expanduser()
    run = json.loads((checkpoint/'run_config.json').read_text())
    config = checkpoint_config(run['model'])
    config.update(model_name_or_path=str(checkpoint),
                  text_model=str(external_models/'t5-base'), local_files_only=True,
                  attn_implementation=attn_implementation, ac_vjepa_encoder_ckpt=None,
                  wan_checkpoint=str(external_models/'Wan2.2_VAE.pth'))
    cfg = ModelConfig.model_validate(config)
    logging.info('Constructing checkpoint architecture: BF16 Qwen, FP32 predictor/frame encoder/T5; attention=%s', attn_implementation)
    backbone_cfg = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    backbone = Qwen3VLForConditionalGeneration._from_config(
        backbone_cfg, attn_implementation=attn_implementation, dtype=torch.bfloat16)
    model = ACVJEPAModel(model_cfg=cfg, vlm_model=backbone)
    logging.info('Loading trained weights directly from %s', checkpoint)
    weights = load_file(str(checkpoint/'model.safetensors'))
    # Safetensors omits duplicate names of tied parameters. Restore only aliases
    # proven to reference the same parameter, then require a strict load.
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    restored = []
    for names in aliases.values():
        present = next((name for name in names if name in weights), None)
        if present:
            for name in names:
                if name not in weights:
                    weights[name] = weights[present]
                    restored.append(name)
    model.load_state_dict(weights, strict=True)
    del weights
    model = model.to(device=device).eval()
    field = run['data']['transform']['field']
    stats = json.loads((external_models/'stats'/Path(field['stat_path']).name).read_text())[field['stat_action_key']]
    metadata = {'model':cfg.model_dump(),
        'preprocessing':{'history_size':1,'image_field':'image','vlm_size':[270,480],
                         'precision':'bfloat16','seed':1039,'require_state':True},
        'normalization':{'action':{'mode':field['action_norm_type'],'eps':field['eps'],**stats}}}
    if field.get('normalize_state', False):
        state_stats = json.loads((external_models/'stats'/Path(field['state_stat_path']).name).read_text())[field['stat_state_key']]
        metadata['normalization']['state'] = {'mode':field['state_norm_type'], 'eps':field['eps'], **state_stats}
    logging.info('Strict checkpoint load passed; restored tied names: %s', restored)
    return model, metadata
