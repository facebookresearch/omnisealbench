#### Advanced usage:

The Omniseal Bench CLI also provides an individual way to watermark content using the registered models, detect the embedded watermarks, and report various metrics. There are two main commands: 

1. `generate_watermark` watermarks content and evaluates the quality of it:

    ```bash
    omniseal eval [image|audio|video] generate_watermark [OPTIONS]
    ```

2. `detect_watermark` detects watermarks in, potentially altered, watermarked content and evaluates the robustness of the watermarking methods:

    ```bash
    omniseal eval [image|audio|video] detect_watermarks [OPTIONS]
    ```

Depending on the command, the `[OPTIONS]` vary. To learn which options are available, use `omniseal eval [image|audio|video] --help`.

### Example 1: Watermarking images

The example below shows how to watermark all images in an image folder using [TrustMark](https://github.com/adobe/trustmark) and how to compute the quality metrics for the generated images:

```bash
omniseal eval image generate_watermarka
        --dataset_dir=./tests/images # TODO: Add a test image into git.
        --result_dir=./results
        --metrics=all
        --seed=42
        --model=trustmark
        # + possible model-dependent parameters

# Output:
# TODO: Add the correct output so the user can check it matches.
```

To list all model-dependent parameters, see the detailed [documentation](docs/README.md) or run `omniseal help [model]`, e.g., `omniseal help trustmark`.

### Example 2: Detect image watermarks

The example below shows how we apply different attacks and run a detector of Trustmark to detect the watermarked results computed in the example 1. The final results consist of
quality metric for each attacked files, together with detection scores indicating the robustness of the models with respect to the corresponding attacks:

```bash
omniseal eval image detect_watermark
        --dataset_dir=./tests/images # TODO: Really needed ???
        --result_dir=./results
        --metrics=all
        --attacks=all # TODO: This seems to do nothing!
        --model=trustmark
        # + possible model-dependent parameters

# Output:
# TODO: Add the correct output so the user can check it matches.
```

To list all available metrics and attacks, see the detailed [documentation](docs/README.md).


### Example 3: Watermarking audio

```bash
omniseal eval audio generate_watermark
        --dataset_type=hf.  # HuggingFace dataset
        --dataset_name=facebook/voxpopuli
        --dataset_hf_subset=en
        --datataset_split=test
        --result_dir=/tmp/tuantran/voxpopuli_audioseal
        --padding=fixed
        --max_length=160000
        --sample_rate=16000
        --message_size=16
        --batch_size=16
        --metrics=all
        --overwrite
        --model=audioseal
        --model_card_or_path=audioseal_wm_16bits
        --num_samples 1000
```

Quick explanation:

- `batch_size`: How many audios are grouped in a batch to speed up computation
- `padding`: How audios in the batch should be padded. Accepted values:
  - `longest`: Pad to the longest samples in the batch
  - `fixed`: Pad to a fixed length for all audios
  - `even`: All samples are asummed to have the same lengths. The batch will simply stack samples into one tensor of dimension `B x C x T`
  - `uneven`: Does not do padding. The batch is a list of tensors
- `max_length`: Only used with padding="fixed". This is the length in which all samples in the batch will be padded / truncated to. In the example, we set max length to 10 seconds for a video of sample rate 16 kHz
- `sample_rate`: Sample rate for the output audios

> [!Tip]
> The argument `num_samples` is optional, used to limit the generation and evaluation to the first 1000 audios. If omitted, all audios in the dataset will be generated.


### Example 4: Detect audio watermarks

```bash
omniseal eval audio detect_watermark
        --input_dir=/tmp/tuantran/voxpopuli_audioseal
        --result_dir=/tmp/tuantran/voxpopuli_audioseal_detection
        --padding=fixed
        --max_length=160000
        --sample_rate=16000
        --metrics=all
        --overwrite
        --attacks additive_gaussian_noise,white_noise
        --model=audioseal_detector
        --model_card_or_path=audioseal_detector_16bits
        --detection_bits=0
```

In the above example, we select specific attacks to perform the detection and robustness evaluation. If all attacks are desired, use `attacks=all` as Example 2.