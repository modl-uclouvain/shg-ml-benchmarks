# Darwin 1.5

[Darwin-1.5](https://github.com/MasterAI-EAM/Darwin) is a Llama-7B variant fine-tuned for materials science question-and-answering.
It has been trained on a corpus of 6M scientific papers in materials science,
distilled into 300k SciQA pairs (plus 300k general QA pairs), providing a 5%
uplift vs LLama-7B.
An additional 8% uplift is achieved by trainong on 21 tabular datasets from
various subdomains.

The model weights can be found linked from the base repo.

## SHG benchmarks

SHG benchmarks were run on a NVIDIA RTX 4080 (16 GB).
