#!/usr/bin/env python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
# lib
from models import GlueNote, GlueHead, LitGlueNoteHead
from eval import TestAlignmentModule
from configs import (Config, 
                     CONFIG_DEFAULTS_LARGE, 
                     CONFIG_DEFAULTS_MID,
                     CONFIG_DEFAULTS_SMALL)


if __name__ == "__main__":

    config = Config(CONFIG_DEFAULTS_MID)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(os.path.dirname(os.getcwd()),"data", "nasap")
    testing_dir = os.path.join(os.path.dirname(os.getcwd()),"data", "testing")
    checkpoints_dir = os.path.join(os.path.dirname(os.getcwd()),"data", "checkpoints")

    test_module = TestAlignmentModule(dir = testing_dir,
                                      log_to_wandb = False)

    # MODEL + TRAINING                
    model = GlueNote(device=DEVICE, 
                     position_number=config.sequence_len,
                     dim_model=config.transformer_dim,
                     num_decoder_layers = config.transformer_blocks,
                     num_heads=config.num_heads,
                     dropout_p=config.dropout,
                     using_decoder=True)
    model.to(DEVICE)
    
    head_model = GlueHead(device=DEVICE, 
                            position_number=config.sequence_len,
                            match_number=config.sequence_len + 1,
                            dim_model=config.transformer_dim,
                            num_encoder_layers = config.decoder_blocks,
                            num_heads=config.num_heads,
                            dropout_p=config.dropout)
    head_model.to(DEVICE)

    chp_large = "final_head_runs/head_large_epoch=1546-step=190281.ckpt"
    chp_mid = "lightning_logs/version_20/checkpoints/epoch=942-step=58466.ckpt"
    chp_small = "final_head_runs/head_small_2_epoch=2004-step=82205.ckpt"
    head_chkpt = os.path.join(checkpoints_dir, chp_mid)
    lit_decoder_model = LitGlueNoteHead.load_from_checkpoint(gluenote = model,
                                                            gluehead = head_model,
                                                            test_module = test_module,
                                                            checkpoint_path = head_chkpt)

    model.to(DEVICE)
    head_model.to(DEVICE)
    test_module.model_tester(model, 
                             head_model, 
                             matching_type = "dtw", # type of postprocessing to be used "dtw", "matrix", "head"
                             log_parangonada = False, # create files ready for parangonada from the predicted alignments
                             log_json = False) # create a JSON with the alignment figures of merit. pass a file name

