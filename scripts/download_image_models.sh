#!/bin/bash

OMNISEAL_CHECKPOINTS="${OMNISEAL_CHECKPOINTS:-$HOME/.cache/omnisealbench/checkpoints}"
echo "Using OMNISEAL_CHECKPOINTS: ${OMNISEAL_CHECKPOINTS}"

# Allow comma-separated model names in OMNISEAL_MODEL to avoid downloading everything.
IFS=',' read -ra REQUESTED_MODELS <<< "${OMNISEAL_MODEL}"
should_download() {
    local model="$1"
    if [ -z "${OMNISEAL_MODEL}" ]; then
        return 0
    fi
    for requested in "${REQUESTED_MODELS[@]}"; do
        # Trim any accidental whitespace around the requested model name.
        requested="${requested//[[:space:]]/}"
        if [ "${requested}" == "${model}" ]; then
            return 0
        fi
    done
    return 1
}

if should_download "wam"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/wam/"
    if [ ! -f "${OMNISEAL_CHECKPOINTS}/wam/wam_mit.pth" ]; then
        wget https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth -P "${OMNISEAL_CHECKPOINTS}/wam/"
    else
        echo "Skipping wam_mit.pth, already exists."
    fi
fi

if should_download "fnns"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/fnns/"
    if [ ! -f "${OMNISEAL_CHECKPOINTS}/fnns/decoder_48b.torchscript.pt" ]; then
        wget https://dl.fbaipublicfiles.com/ssl_watermarking/decoder_48b.torchscript.pt -P "${OMNISEAL_CHECKPOINTS}/fnns/"
    else
        echo "Skipping decoder_48b.torchscript.pt in fnns, already exists."
    fi
fi

if should_download "hidden"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/hidden/"
    if [ ! -f "${OMNISEAL_CHECKPOINTS}/hidden/encoder_with_jnd_48b.torchscript.pt" ]; then
        wget https://dl.fbaipublicfiles.com/ssl_watermarking/encoder_with_jnd_48b.torchscript.pt -P "${OMNISEAL_CHECKPOINTS}/hidden/"
    else
        echo "Skipping encoder_with_jnd_48b.torchscript.pt in hidden, already exists."
    fi
    if [ ! -f "${OMNISEAL_CHECKPOINTS}/hidden/decoder_48b.torchscript.pt" ]; then
        wget https://dl.fbaipublicfiles.com/ssl_watermarking/decoder_48b.torchscript.pt -P "${OMNISEAL_CHECKPOINTS}/hidden/"
    else
        echo "Skipping decoder_48b.torchscript.pt in hidden, already exists."
    fi
fi

if should_download "ssl"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/ssl/"
    if [ ! -f "${OMNISEAL_CHECKPOINTS}/ssl/dino_r50_90_plus_w.torchscript.pt" ]; then
        wget https://dl.fbaipublicfiles.com/ssl_watermarking/dino_r50_90_plus_w.torchscript.pt -P "${OMNISEAL_CHECKPOINTS}/ssl/"
    else
        echo "Skipping dino_r50_90_plus_w.torchscript.pt in ssl, already exists."
    fi
fi

if should_download "cin"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/cin/"

    if [ ! -f "${OMNISEAL_CHECKPOINTS}/cin/opt.yml" ]; then
        wget https://raw.githubusercontent.com/rmpku/CIN/refs/heads/main/codes/options/opt.yml -O "${OMNISEAL_CHECKPOINTS}/cin/opt.yml"
    else
        echo "Skipping opt.yml in cin, already exists."
    fi

    if [ ! -f "${OMNISEAL_CHECKPOINTS}/cin/cinNet&nsmNet.pth" ]; then
        gdown --fuzzy https://drive.google.com/file/d/1wqnqhPv92mHwkEI4nMh-sI5aDgh-usr7/view?usp=share_link -O "${OMNISEAL_CHECKPOINTS}/cin/cinNet&nsmNet.pth"
    else
        echo "Skipping cinNet&nsmNet.pth in cin, already exists."
    fi
fi

if should_download "cin_jit"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/cin_jit/"

    if [ ! -f "${OMNISEAL_CHECKPOINTS}/cin_jit/cin_nsm_decoder.pt" ] || [ ! -f "${OMNISEAL_CHECKPOINTS}/cin_jit/cin_nsm_encoder.pt" ]; then
        echo "Generating CIN JIT models..."
        python scripts/generate_cin_jit.py --export_dir cin_jit
    else
        echo "Skipping cin_nsm_decoder and cin_nsm_encoder in cin_jit, already exists."
    fi
fi

if should_download "mbrs"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/mbrs/"

    if [ ! -f "${OMNISEAL_CHECKPOINTS}/mbrs/EC_42.pth" ]; then
        gdown https://drive.google.com/uc?id=13R8F_DsmC7firokQ5lSBTkgiuK1naryc -O "${OMNISEAL_CHECKPOINTS}/mbrs/EC_42.pth" # EC_42.pth
    else
        echo "Skipping EC_42.pth in mbrs, already exists."
    fi
fi

if should_download "mbrs_jit"; then
    mkdir -p "${OMNISEAL_CHECKPOINTS}/mbrs_jit/"

    # Download MBRS first if not already present.
    if [ ! -f "${OMNISEAL_CHECKPOINTS}/mbrs/EC_42.pth" ]; then
        echo "download EC_42"
        gdown https://drive.google.com/uc?id=13R8F_DsmC7firokQ5lSBTkgiuK1naryc -O "${OMNISEAL_CHECKPOINTS}/mbrs/EC_42.pth" # EC_42.pth
    else
        echo "Skipping EC_42.pth in mbrs, already exists."
    fi

    if [ ! -f "${OMNISEAL_CHECKPOINTS}/mbrs_jit/mbrs_256_m256_encoder.pt" ] || [ ! -f "${OMNISEAL_CHECKPOINTS}/mbrs_jit/mbrs_256_m256_decoder.pt" ]; then
        echo "Generating MBRS JIT models..."
        python scripts/generate_mbrs_jit.py --export_dir mbrs_jit
    else
        echo "Skipping mbrs_256_m256_encoder and mbrs_256_m256_decoder in mbrs_jit, already exists."
    fi
fi

# mkdir -p "${OMNISEAL_CHECKPOINTS}/invismark/"
