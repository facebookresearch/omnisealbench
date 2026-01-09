# export OMNISEAL_CHECKPOINTS=/path/to/the/checkpoints

from pathlib import Path

import torch

from omnisealbench.config import OMNISEAL_CHECKPOINTS_DIR
from omnisealbench.models.mbrs_src import Network


def generate_mbrs_jit(export_dir: str) -> None:
    device = torch.device("cpu")

    assert Path(OMNISEAL_CHECKPOINTS_DIR).exists(), f"{OMNISEAL_CHECKPOINTS_DIR} does not exist"

    model_card_or_path = Path(OMNISEAL_CHECKPOINTS_DIR)/"mbrs/EC_42.pth"
    assert model_card_or_path.exists(), f"{model_card_or_path} does not exist"

    nbits = 256
    H, W  = 256, 256

    model = Network(H, W, nbits, with_diffusion=False)
    model.load_model_ed(model_card_or_path)
    model.encoder_decoder.to(device).eval()

    export_dir = Path(OMNISEAL_CHECKPOINTS_DIR)/export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    m = torch.jit.script(model.encoder_decoder.encoder.eval())
    torch.jit.save(m, export_dir/'mbrs_256_m256_encoder.pt')

    m = torch.jit.script(model.encoder_decoder.decoder.eval())
    torch.jit.save(m, export_dir/'mbrs_256_m256_decoder.pt')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate TorchScript JIT models for MBRS encoder and decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--export_dir",
        type=str,
        default="mbrs_jit",
        help="Directory name to export the JIT models to (relative to OMNISEAL_CHECKPOINTS)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_mbrs_jit(export_dir=args.export_dir)
        print("✓ MBRS JIT model generation completed successfully!")
    except Exception as e:
        print(f"✗ Error generating MBRS JIT models: {e}")
        exit(1)
