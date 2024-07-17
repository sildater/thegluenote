
#!/usr/bin/env python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
# torch
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
# lib
from datasets import AlignmentDataset
from models import (GlueNote,  
                    GlueHead, 
                    LitGlueNoteHead)
from eval import TestAlignmentModule
from configs import (Config, 
                    CONFIG_DEFAULTS_LARGE, 
                    CONFIG_DEFAULTS_MID, 
                    CONFIG_DEFAULTS_SMALL)
# sweeps
import wandb

if __name__ == "__main__":

    # WANDB Config: if fine-tuning, set the config corresponding to the checkpoint
    wandb.init(config=CONFIG_DEFAULTS_MID, project="thegluenote", name="run_0")
    config = wandb.config
    # WANDB Logging
    display_name =  "dim_" + str(config.transformer_dim) + \
                    "_seql_" + str(config.sequence_len) + \
                    "_bs_" + str(config.batch_size) + \
                    str(np.random.randint(100,1000))

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.path.dirname(os.getcwd()),"data", "nasap")
    testing_dir = os.path.join(os.path.dirname(os.getcwd()),"data", "testing")
    checkpoints_dir = os.path.join(os.path.dirname(os.getcwd()),"data", "checkpoints")

    # datasets
    full_dataset = AlignmentDataset(data_dir,
                                    notes_per_sample = config.sequence_len,
                                    prob_deletions=config.prob_del,
                                    prob_insertions=config.prob_ins,
                                    prob_repeats=config.prob_rep,
                                    prob_skips=config.prob_ski)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, 
                                                        [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.batch_size, 
                                  shuffle=True,
                                  collate_fn=full_dataset.collate,
                                  num_workers = 8)
    val_dataloader = DataLoader(val_dataset, 
                                  batch_size=config.batch_size, 
                                  shuffle=False,
                                  collate_fn=full_dataset.collate,
                                  num_workers = 8)
    
    test_module = TestAlignmentModule(
        tokenizer = full_dataset.tokenizer,
        dir = testing_dir,
        log_to_wandb = True)

    # MODEL + TRAINING
    trainer = pl.Trainer(max_epochs=config.epochs,
                         devices=1, 
                         num_nodes=1,
                         default_root_dir=checkpoints_dir,
                         enable_checkpointing=True)
                       
    model = GlueNote(device=DEVICE, 
                     position_number=config.sequence_len,
                     dim_model=config.transformer_dim,
                     num_decoder_layers = config.transformer_blocks,
                     num_heads=config.num_heads,
                     dropout_p=config.dropout,
                     using_decoder = True)
    model.to(DEVICE)

    decoder_model = GlueHead(device=DEVICE, 
                            position_number=config.sequence_len,
                            match_number=config.sequence_len + 1,
                            dim_model=config.transformer_dim,
                            num_encoder_layers = config.decoder_blocks,
                            num_heads=config.num_heads,
                            dropout_p=config.dropout)
    decoder_model.to(DEVICE)


    # LITMODEL - initialize and train from scratch
    # lit_decoder_model = LitGlueNoteHead(gluenote = model,
    #                                     gluehead=decoder_model,
    #                                     lr = config.learning_rate,
    #                                     test_module = test_module,
    #                                     log_test_interval = 1)
    
    chkpt = os.path.join(checkpoints_dir, "path/to/prev/checkpoint.ckpt")

    lit_decoder_model = LitGlueNoteHead.load_from_checkpoint(gluenote = model, 
                                                             gluehead=decoder_model,
                                                             test_module = test_module,
                                                             checkpoint_path = chkpt,
                                                             lr = config.learning_rate,
                                                             log_test_interval = 1)

    trainer.fit(model=lit_decoder_model, 
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    