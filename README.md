# OmniSealBench

This repository provides a comprehensive benchmark for evaluating the performance of neural watermarking techniques. The benchmark includes a variety of datasets, evaluation metrics, and tools for training and testing neural networks for watermarking.


## Documentation

For detailed information about the models and metrics used in OmniSealBench, please refer to the [Documentation](omnisealbench/docs/README.md) section. This section contains markdown files that describe each model (and how to download it) and metric in detail.


## 🔥 Quick Start

> [!Tip]
>
> OmniSeal ❤️ [AudioMarkBench](https://github.com/moyangkuo/AudioMarkBench)!
> RAVDESS has been integrated to omnisealbench datasets!

To quickly perform watermarking generation and evaluation on RAVDESS:

```bash
pip install omnisealbench --upgrade
omnisealbench.evaluation --eval_type [image|audio] # required to specify the evaluation modality
                         --config "config file path.yaml" # optional, by default it is using the existing configuration
                         --model "config file path.yaml" or name from the datacards/models folder # by default it evalues all the registered models
                         --dataset "config file path.yaml" or name from the datacards/datasets folder # by default it evalues all the registered datasets
                         --num_workers 4 # this allows a faster 'effects + watermarking detection' processing
                         --batch_size 4 # faster parallel decoding by using batching
                         --dataset_dir "/path/to/audio_files/directory" # when using 'local' dataset implementation we can specify a path to audio files
                         --save_ids 0,1,3-5 # generates some example of attacking/watermarking results for specified image (image only for the moment) indices
```

### Audio models

- `AudioSeal': Proactive Localized Watermarking:

```bash
omnisealbench.evaluate --model "audioseal"        \
                       --etc...
```
You can checkout the generation at `~/.cache/omnisealbench/watermarking/[ravdess|custom]_audioseal` and the results at `./eval_results.json`


- `Wavmark': AI-based Audio Watermarking Tool:

```bash
omnisealbench.evaluate --model "wavmark"        \
                       --etc...
```
You can checkout the generation at `~/.cache/omnisealbench/watermarking/[ravdess|custom]_wavmark` and the results at `./eval_results.json`



### Image models

- `WAM': (watermark-anything models) model that combines an embedder, a detector, and an augmenter:

```bash
omnisealbench.evaluate --eval_type image \
                       --model "wam" \
                       --dataset "coco" \
                       --dataset_dir "/path/to/COCO/val2014" \
                       --num_samples 128 \
                       --num_workers 2 \
                       --batch_size 2 \
                       --save_ids 0,1,3-5
```
You can checkout the generation at `~/.cache/omnisealbench/watermarking/val2014_wam` and the results at `./eval_results.json`.
The examples for the specified ids are in `./examples/{attack}/*.png`.



<details><summary>⏬ Install nightly version <i>:: click to expand ::</i></summary>
<div>

```bash
pip install --upgrade "git+https://github.com/facebookresearch/omnisealbench.git"                           # all modalities
pip install --upgrade "omnisealbench[audio] @ git+https://github.com/facebookresearch/omnisealbench@master" # only audio
```

</div>
</details>

<details><summary>⏬ Using OmniSealBench as a local repo? <i>:: click to expand ::</i></summary>
<div>

```bash
git clone https://github.com/facebookresearch/omnisealbench.git
cd omnisealbench
export PYTHONPATH=$PYTHONPATH:$(pwd)
pip install -e .

```

</div>
</details>


## License
This repository is licensed under CC-BY-NC 4.0. See LICENSE.md file for further details.

## 🙏 Acknowledgement

- [AudioMarkBench](https://github.com/moyangkuo/AudioMarkBench)
