#!/bin/bash

OMNISEAL_CHECKPOINTS="${OMNISEAL_CHECKPOINTS:-$HOME/.cache/omnisealbench/checkpoints}"
echo "Using OMNISEAL_CHECKPOINTS: ${OMNISEAL_CHECKPOINTS}"

mkdir -p "${OMNISEAL_CHECKPOINTS}/rivagan/"
mkdir -p "${OMNISEAL_CHECKPOINTS}/rivagan/product.m00-n100-1554568357"

if [ ! -f "${OMNISEAL_CHECKPOINTS}/rivagan/product.m00-n100-1554568357/model.pt" ]; then
    wget https://raw.githubusercontent.com/DAI-Lab/RivaGAN/refs/heads/paper_results/results/product.m00-n100-1554568357/model.pt -P "${OMNISEAL_CHECKPOINTS}/rivagan/product.m00-n100-1554568357/"
else
    echo "Skipping rivagan product.m00-n100-1554568357/model.pt, already exists."
fi