
# RuBioRoBERTa

The models can be found here:

https://huggingface.co/alexyalunin/RuBioRoBERTa

https://huggingface.co/alexyalunin/RuBioBERT


## Reproduce
1. Download RuMedBench and install `requirements.txt`

2. For pre-training run `scripts/1_pretrain.sh` or `scripts/1_pretrain_distributed.sh`

3. For fine-tuning on RuMedBench run `scripts/2_RuMedBench.sh`

## Contact
If you have any questions, please post a Github issue.

## Citation
If you find this repo useful, please cite as:
```
@article{yalunin2022rubioroberta,
  title={RuBioRoBERTa: a pre-trained biomedical language model for Russian language biomedical text mining},
  author={Yalunin, Alexander and Nesterov, Alexander and Umerenkov, Dmitriy},
  journal={arXiv preprint arXiv:2204.03951},
  year={2022}
}

```
