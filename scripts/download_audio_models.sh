#!/bin/bash

OMNISEAL_CHECKPOINTS="${OMNISEAL_CHECKPOINTS:-$HOME/.cache/omnisealbench/checkpoints}"
echo "Using OMNISEAL_CHECKPOINTS: ${OMNISEAL_CHECKPOINTS}"

mkdir -p "${OMNISEAL_CHECKPOINTS}/timbre/"
if [ ! -f "${OMNISEAL_CHECKPOINTS}/timbre/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar" ]; then
    gdown https://drive.google.com/uc?id=13R6NxhKP_vR2qjPJy9QjiSRQEbzwiZEF -O "${OMNISEAL_CHECKPOINTS}/timbre/compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar"
else
    echo "Skipping compressed_none-conv2_ep_20_2023-02-14_02_24_57.pth.tar in timbre, already exists."
fi