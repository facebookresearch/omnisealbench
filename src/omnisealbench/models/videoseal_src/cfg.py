# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from typing import Optional
import omegaconf
import torch
import torch.distributed as dist
import videoseal
import videoseal.utils.dist as udist
from omegaconf import DictConfig, OmegaConf
from videoseal.augmentation.augmenter import get_dummy_augmenter
from videoseal.models import build_baseline, build_embedder, build_extractor

try:
    # try loading videoseal-public first
    from videoseal.models import Videoseal
    from videoseal.utils.cfg import VideosealConfig

except ImportError as ex:
    if "videoseal" not in str(ex):
        raise ex
    # Use fairinternal videoseal
    from videoseal.models import VideoWam as Videoseal
    from videoseal.utils.cfg import VideoWamConfig as VideosealConfig

from videoseal.modules.jnd import JND
from videoseal.utils.cfg import (SubModelConfig, is_url,
                                 maybe_download_checkpoint)

DEFAULT_CARD = 'videoseal_1.0'


def get_config_from_checkpoint(ckpt_path: Path) -> VideosealConfig:
    """
    Load configuration from a checkpoint file.

    Args:
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    VideosealConfig: Loaded configuration.
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    args = checkpoint['args']
    args = OmegaConf.create(args)

    if not isinstance(args, DictConfig):
        raise Exception("Expected logfile to contain params dictionary.")
    
    # Order of config loading:
    # 1. From the working directory
    # 2. From the checkpoint directory
    # 3. From the source code directory of videoseal
    
    embedder_cfg_path = Path(args.embedder_config)
    if not embedder_cfg_path.is_file():
        embedder_cfg_path = Path(ckpt_path).parent.joinpath(args.embedder_config)
    if not embedder_cfg_path.is_file():
        embedder_cfg_path = Path(videoseal.__file__).parent.parent.joinpath(args.embedder_config)

    extractor_cfg_path = Path(args.extractor_config)
    if not extractor_cfg_path.is_file():
        extractor_cfg_path = Path(ckpt_path).parent.joinpath(args.extractor_config)
    if not extractor_cfg_path.is_file():
        extractor_cfg_path = Path(videoseal.__file__).parent.parent.joinpath(args.extractor_config)

    # Load sub-model configurations
    embedder_cfg = OmegaConf.load(embedder_cfg_path)
    extractor_cfg = OmegaConf.load(extractor_cfg_path)

    # Create sub-model configurations
    embedder_model = args.embedder_model or embedder_cfg.model
    embedder_params = embedder_cfg[embedder_model]
    extractor_model = args.extractor_model or extractor_cfg.model
    extractor_params = extractor_cfg[extractor_model]

    return VideosealConfig(
        args=args,
        embedder=SubModelConfig(model=embedder_model, params=embedder_params),
        extractor=SubModelConfig(model=extractor_model, params=extractor_params),
    )


def setup_model(config: VideosealConfig, ckpt_path: Path) -> Videoseal:
    """
    Set up a Video Seal model from a configuration and checkpoint file.

    Args:
    config (VideosealConfig): Model configuration.
    ckpt_path (Path): Path to the checkpoint file.

    Returns:
    Videoseal: Loaded model.
    """
    args = config.args

    # prepare some args for backward compatibility
    if "img_size_proc" in args:
        args.img_size = args.img_size_proc
    else:
        args.img_size = args.img_size_extractor

    if "hidden_size_multiplier" in args:
        args.hidden_size_multiplier = args.hidden_size_multiplier
    else:
        args.hidden_size_multiplier = 2

    # Build models
    embedder = build_embedder(config.embedder.model, config.embedder.params, args.nbits, args.hidden_size_multiplier)
    extractor = build_extractor(config.extractor.model, config.extractor.params, args.img_size, args.nbits)
    augmenter = get_dummy_augmenter()  # does nothing

    # Build attenuation
    if args.attenuation.lower().startswith("jnd"):
        
        # Load attenuation config from:
        # 1. working directory
        # 2. checkpoint directory
        # 3. source code directory of videoseal
        
        attenuation_cfg_path = Path(args.attenuation_config)
        if not attenuation_cfg_path.is_file():
            attenuation_cfg_path = Path(ckpt_path).parent.parent.joinpath("code", args.attenuation_config)
        if not attenuation_cfg_path.is_file():
            attenuation_cfg_path = Path(videoseal.__file__).parent.parent.joinpath("configs", "attenuation.yaml")
        
        attenuation_cfg = omegaconf.OmegaConf.load(attenuation_cfg_path)
        attenuation = JND(**attenuation_cfg[args.attenuation])
    else:
        attenuation = None

    # Build the complete model
    wam = Videoseal(
        embedder,
        extractor,
        augmenter,
        attenuation=attenuation,
        scaling_w=args.scaling_w,
        scaling_i=args.scaling_i,
        img_size=args.img_size,
        chunk_size=args.get("videoseal_chunk_size", 32),
        step_size=args.get("videoseal_step_size", 4),
    )

    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        msg = wam.load_state_dict(checkpoint['model'], strict=False)
        # print(f"Model loaded successfully from {ckpt_path} with message: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt_path}")

    return wam

def setup_model_from_checkpoint(ckpt_path: str, attenuation: Optional[str] = None, attenuation_config: Path | str = None) -> Videoseal:
    """
    # Example usage
    ckpt_path = '/path/to/videoseal/checkpoint.pth'
    wam = setup_model_from_checkpoint(ckpt_path)

    or 
    ckpt_path = 'baseline/wam'
    wam = setup_model_from_checkpoint(ckpt_path)
    """
    # load baselines. Should be in the format of "baseline/{method}"
    ckpt_path = str(ckpt_path)
    if "baseline" in ckpt_path:
        method = ckpt_path.split('/')[-1]
        return build_baseline(method)
    # load videoseal model card
    elif ckpt_path.startswith('videoseal'):
        assert attenuation is not None, "attenuation must be provided when loading from model card"
        return setup_model_from_model_card(ckpt_path, attenuation=attenuation, attenuation_config=attenuation_config)
    # load videoseal ckpts
    else:
        config = get_config_from_checkpoint(ckpt_path)
        return setup_model(config, ckpt_path)


def setup_model_from_model_card(model_card: Path | str, attenuation: str, attenuation_config: Path | str) -> Videoseal:
    """
    Set up a Video Seal model from a model card YAML file.
    Args:
        model_card (Path | str): Path to the model card YAML file or name of the model card.
    Returns:
        Videoseal: Loaded model.
    """
    cards_dir = Path(videoseal.__file__).parent / "cards"

    if model_card == 'videoseal':
        model_card = DEFAULT_CARD

    if isinstance(model_card, str):
        available_cards = [card.stem for card in cards_dir.glob('*.yaml')]
        if model_card not in available_cards:
            print(f"Available model cards: {', '.join(available_cards)}")
            raise FileNotFoundError(f"Model card '{model_card}' not found in {cards_dir}")
        model_card_path = cards_dir / f'{model_card}.yaml'
    elif isinstance(model_card, Path):
        if not model_card.exists():
            print(f"Available model cards: {', '.join([card.stem for card in cards_dir.glob('*.yaml')])}")
            raise FileNotFoundError(f"Model card file '{model_card}' not found")
        model_card_path = model_card
    else:
        raise TypeError("Model card must be a string or a Path object")

    with open(model_card_path, 'r', encoding='utf-8') as file:
        config = OmegaConf.load(file)

        # Override attenuation
        if attenuation:
            config.args.attenuation = attenuation
        if attenuation_config:
            config.args.attenuation_config = str(attenuation_config)

    if Path(config.checkpoint_path).is_file():
        ckpt_path = Path(config.checkpoint_path)

    elif str(config.checkpoint_path).startswith("https://huggingface.co/facebook/video_seal/"):
        # Extract the filename from the URL
        import os
        checkpoint_url = str(config.checkpoint_path)

        # Handle URLs with or without 'resolve/main'
        if "/resolve/" in checkpoint_url:
            fname = os.path.basename(checkpoint_url.split("/resolve/", 1)[1])  # Extract after 'resolve/<branch>/'
        else:
            fname = os.path.basename(checkpoint_url)  # Extract the filename directly

        try:
            from huggingface_hub import hf_hub_download
        except ModuleNotFoundError:
            print(
                f"The model path {config.checkpoint_path} seems to be a direct HF path, "
                "but you do not have `huggingface_hub` installed. Install it with "
                "`pip install huggingface_hub` to use this feature."
            )
            raise

        # Download the checkpoint
        ckpt_path = hf_hub_download(
            repo_id="facebook/video_seal",  # The repository ID
            filename=fname  # Dynamically determined filename
        )

    elif is_url(config.checkpoint_path):
        if udist.is_dist_avail_and_initialized():
            # download only on the main process
            if udist.is_main_process():
                ckpt_path = maybe_download_checkpoint(config.checkpoint_path)
            dist.barrier()

        ckpt_path = maybe_download_checkpoint(config.checkpoint_path)
    else:
        raise RuntimeError(f"Path or uri {config.checkpoint_path} is unknown or does not exist")

    return setup_model(config, ckpt_path)
