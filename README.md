# TheGlueNote

TheGlueNote is representation model for note-wise music alignment.

The repo structure is as follows:
- data contains:
    - `checkpoints`:  small and mid model, large model [available online](https://cloud.cp.jku.at/index.php/s/em5k9FYfn5grf9A)
    - `nasap`: raw training data based on the MIDI files in the [(n)ASAP dataset](https://github.com/CPJKU/asap-dataset)
    - `testing`: Vienna4x22 data for testing as well as output directories for images and [parangonada files](https://sildater.github.io/parangonada/)
- src contains:
    - `config`: configurations file
    - `datasets`: pytorch dataset, collator, and data augmentation (synthetic complex mismatches)
    - `eval`: testing utility
    - `model`: model source code
    - `post`: post-processing of model predictions
    - `test.py` : run pre-trained models on the 4x22 data using the testing utility
    - `train.py` : train pre-trained or fresh models on the (n)ASAP data using the synthetic augmentations


## Installation

clone the repository and make sure you have some dependencies installed (easiest via the `requirements.txt`, except for pytorch which needs to fit your machine):
- pytorch
- pytorch-lightning
- wandb
- numpy
- scipy
- miditok
- symusic
- parangonar 
- matplotlib

## Usage

This repository is collects data, code, and checkpoints as used for the publication below. 
The provided scripts allow for testing and (re-)training in a very similar fashion to what was done for the experiments in the paper.

If you want to use TheGlueNote with as little hassle as possible, it is also integrated in [parangonar](https://github.com/sildater/parangonar), 
a python library for note alignment that is pip-installable and provides a simple interface to apply several note alignment algorithms.

## Cite us

```bibtex
@inproceedings{peter24thegluenote,
  title={TheGlueNote: Learned Representations for Robust and Flexible Note Alignment},
  author={Peter, Silvan David and Widmer, Gerhard},
  booktitle={Proceedings of the 25th International Society for Music Information Retrieval Conference (ISMIR), San Francisco, USA},
  year={2024}
}
```
