# Sources

Model and pipeline code were written for this project. Models and BPE were trained
from random initialization; pretrained weights and tokenizers were not used.

Design references: [nanoGPT](https://github.com/karpathy/nanoGPT)
([MIT](https://github.com/karpathy/nanoGPT/blob/master/LICENSE)),
[MobileLLM](https://github.com/facebookresearch/MobileLLM), and
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-360M).
These runs are adaptations, not reproductions of the published training recipes.

Corpora: [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
and [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia).
Source revisions and document provenance are retained in data manifests.
See the original dataset cards for attribution and licensing.
