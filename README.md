# hf-tokenizers-experiments
Experiments with [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers).

![hf-experiments-128](https://user-images.githubusercontent.com/163333/117465228-c529e100-af51-11eb-92c4-2dca58b8f0f9.png)

## On 🔥 🔬 Experiments :new:
Microsoft Multilingual MiniLM 12-layer, 384-hidden, 12-heads, 21M Transformer parameters, 96M embedding parameters plus SentencePiece BPE Tokenizer. MiniLM is a distilled model from the paper ["MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers"](https://arxiv.org/abs/2002.10957)

## 🤗 Huggingface 🔬 Tokenizers Experiments
Currently we provide customization of the following tokenizers

- SentencePiece BPE Tokenizer :new: 🔥


and vocabularies (vocabulary and merge files) for the folliwing models

- [Microsoft Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)

## How to install
```
git clone https://github.com/loretoparisi/hf-tokenizers-experiments
cd hf-tokenizers-experiments
npm install
```

## How to run
```
cd hf-tokenizers-experiments
cd examples/
node minilm-tokenizer
```
