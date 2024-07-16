import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
import lightning.pytorch as pl
import wandb
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class GlueNote(nn.Module):
    """
    GlueNote non-causal transformer encoder
    with learned positional encoding
    """
    # Constructor
    def __init__(
        self,
        device,
        token_number = 314, # tokenizer.vocab, but many are unused
        position_number = 128,
        dim_model = 256,
        dim_feedforward = None,
        num_heads = 8,
        num_decoder_layers = 6,
        dropout_p = 0.1,
        activation = nn.GELU(),
        using_decoder = False
    ):
        super().__init__()
        self.device = device
        self.token_number = token_number
        self.position_number = position_number + 1 # prepended 0
        self.dim_model = dim_model
        self.activation = activation
        if dim_feedforward is not None:
            self.dim_feedforward = dim_feedforward
        else: 
            self.dim_feedforward = self.dim_model * 4
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.using_decoder = using_decoder
        # LAYERS
        self.positions = torch.arange(self.position_number * 2).to(self.device)
        self.positional_encoder = nn.Embedding(self.position_number * 2,self.dim_model) 
        self.embedding = nn.Embedding(self.token_number,self.dim_model) 
        
        # DECODER LAYERS
        D_layers = TransformerEncoderLayer(self.dim_model, 
                                            nhead = self.num_heads, 
                                            dim_feedforward = self.dim_feedforward, 
                                            dropout=self.dropout_p,
                                            norm_first = True,
                                            activation = self.activation)

        self.layer_normalization= LayerNorm(self.dim_model)
        self.layer_normalization_input = LayerNorm(self.dim_model)

        self.transformerDECODER = TransformerEncoder(
            encoder_layer = D_layers,
            num_layers = num_decoder_layers,
            norm = self.layer_normalization,
            enable_nested_tensor=False
        )

        self.mlp1 = nn.Linear(self.dim_model,self.dim_feedforward)
        self.mlp_activation = activation = self.activation
        self.mlp2 = nn.Linear(self.dim_feedforward,self.dim_model)

        self.embed_out = nn.Linear(self.dim_model,self.dim_model)
        
    def forward(self, src):
        # # let's just only use pitch for now, no aggregation
        # src = src[:,1::4]
        src = self.embedding(src)
        
        # AGGREGATING 4 ATTRIBUTES
        eshape = src.shape
        src = torch.sum(src.reshape(eshape[0], -1, 4, eshape[-1]), dim=-2)
        
        # to obtain size (sequence length, batch_size, dim_model)
        src = src.permute(1,0,2)
        # POSITIONAL ENCODING
        pos = self.positional_encoder(self.positions)
        src += pos.unsqueeze(1)
        # src = self.positional_encoder(src)
        src = self.layer_normalization_input(src)
        # Transformer blocks - Out size = (sequence length, batch_size, dim_model)
        transformer_out = self.transformerDECODER(src=src)    
        # mlp_out = self.mlp2(self.melp_activation(self.mlp1(transformer_out))) 
        mlp_out = self.embed_out(transformer_out)
        predictions = torch.einsum('ibk,jbk->bij', mlp_out[:self.position_number,:,:], mlp_out[self.position_number:,:,:])
        if self.using_decoder:
            return predictions, mlp_out
        else:
            return predictions

class GlueHead(nn.Module):
    """
    GlueHead non-causal transformer encoder
    with learned positional encoding
    to predict match sequence from pairwise similairities
    """
    # Constructor
    def __init__(
        self,
        device,
        match_number = 513, # one more than reference seq
        position_number = 128,
        dim_model = 256,
        dim_feedforward = None,
        num_heads = 8,
        num_encoder_layers = 6,
        dropout_p = 0.1,
        activation = nn.GELU(),
    ):
        super().__init__()
        self.device = device
        self.match_number = match_number
        self.position_number = position_number
        self.dim_model = dim_model
        self.activation = activation
        if dim_feedforward is not None:
            self.dim_feedforward = dim_feedforward
        else: 
            self.dim_feedforward = self.dim_model * 4
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dropout_p = dropout_p

        # LAYERS
        self.positions = torch.arange(self.position_number).to(self.device)
        self.positional_encoder = nn.Embedding(self.position_number,self.dim_model) 

        self.embedding_similarity = nn.Linear(self.match_number, self.dim_model)
        self.embedding_pitch_similarity = nn.Linear(self.match_number, self.dim_model)
        
        # ENCODER LAYERS
        layers = TransformerEncoderLayer(self.dim_model, 
                                        nhead = self.num_heads, 
                                        dim_feedforward = self.dim_feedforward, 
                                        dropout=self.dropout_p,
                                        norm_first = True,
                                        activation = self.activation)

        self.layer_normalization= LayerNorm(self.dim_model)
        self.layer_normalization_input = LayerNorm(self.dim_model)

        self.transformerDECODER = TransformerEncoder(
            encoder_layer = layers,
            num_layers = num_encoder_layers,
            norm = self.layer_normalization,
            enable_nested_tensor=False
        )

        self.embed_out = nn.Linear(self.dim_model,self.match_number)
        
    def forward(self, seq_sim, pitch_sim = None):
        # sequence and pitch similarity matrices come in shape
        # (batch_size, match_number, seq_len) 
        # e.g. 2 x 513 x 512
        # permute to seq - batch - match_num
        seq_sim = seq_sim.permute(2,0,1)
        src = self.embedding_similarity(seq_sim) 
        if pitch_sim is not None:
            pitch_sim = pitch_sim.permute(2,0,1)
            src += self.embedding_pitch_similarity(pitch_sim)
        # POSITIONAL ENCODING
        pos = self.positional_encoder(self.positions)
        src += pos.unsqueeze(1)
        # normalize
        src = self.layer_normalization_input(src)
        # Transformer blocks - Out size = (sequence length, batch_size, dim_model)
        transformer_out = self.transformerDECODER(src=src)    
        # (seq x batch x 513)
        logits = self.embed_out(transformer_out)
        # output permute for ce loss: batch x classes x seq
        logits = logits.permute(1, 2, 0)
        return logits

class LitGlueNote(pl.LightningModule):
    def __init__(self, 
                 gluenote,
                 lr = 5e-4,
                 test_module = None,
                 log_test_interval = 10000,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['gluenote', "test_module"])
        self.gluenote = gluenote
        self.lr = lr
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.test_module = test_module
        self.log_test_interval = log_test_interval
        self.val_collector = list()

    def training_step(self, batch, batch_idx):
        # unpack and move
        sequences = batch[0]
        targets = batch[1]
        targets_s1 = batch[2]
        targets_s2 = batch[3]
        sequences = sequences.to(self.gluenote.device)
        targets = targets.to(self.gluenote.device)
        targets_s1 = targets_s1.to(self.gluenote.device)
        targets_s2 = targets_s2.to(self.gluenote.device)

        # compute predictions and loss
        predictions, _ = self.gluenote(sequences)
        # for each column (s2) predict row (s1)
        loss_s2 = self.classification_loss(predictions[:,:,1:], targets_s2[:,1:])
        # swap and for each column (s1) predict row (s2)
        loss_s1 = self.classification_loss(torch.swapaxes(predictions,1,2)[:,:,1:], targets_s1[:,1:])
        loss = loss_s1 + loss_s2
        wandb.log({"train_loss": loss.item(),
                  "s1_loss": loss.item(),
                  "s2_loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        # unpack and move
        sequences = batch[0]
        targets = batch[1]
        targets_s1 = batch[2]
        targets_s2 = batch[3]
        sequences = sequences.to(self.gluenote.device)
        targets = targets.to(self.gluenote.device)
        targets_s1 = targets_s1.to(self.gluenote.device)
        targets_s2 = targets_s2.to(self.gluenote.device)

        # compute predictions and loss
        predictions, _ = self.gluenote(sequences)
        # for each column (s2) predict row (s1)
        loss_s2 = self.classification_loss(predictions[:,:,1:], targets_s2[:,1:])
        # swap and for each column (s1) predict row (s2)
        loss_s1 = self.classification_loss(torch.swapaxes(predictions,1,2)[:,:,1:], targets_s1[:,1:])
        loss = loss_s1 + loss_s2
        wandb.log({"validation_loss": loss.item()})
        self.val_collector.append((predictions, targets))
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
                    }
                }
    
    def on_validation_epoch_end(self, verbose = False):
        try:
            # if (self.current_epoch+1) % self.log_test_interval == 0:
            #     self.test_module.model_tester(self.gluenote, self.current_epoch)
            predictions, targets = self.val_collector[0]
            self.test_module.log_similarity_matrix(predictions[0,:,:], 
                                                    targets[0,:,:],
                                                    self.current_epoch)
            
            self.val_collector = list()
        except:
            print("logging failed at epoch: ", self.current_epoch)

class LitGlueNoteHead(pl.LightningModule):
    def __init__(self, 
                 gluenote,
                 gluehead,
                 lr = 5e-4,
                 test_module = None,
                 log_test_interval = 10000,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['gluenote', "test_module", "gluehead"])
        self.gluenote = gluenote
        self.gluehead = gluehead
        self.lr = lr
        # self.rec_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.test_module = test_module
        self.log_test_interval = log_test_interval
        self.val_collector = list()
        self.train_loss_collector = list()
        self.val_loss_collector = list()
        self.val_acc_collector = list()

    def training_step(self, batch, batch_idx):
        # unpack and move
        sequences = batch[0]
        targets = batch[1]
        targets_s1 = batch[2]
        targets_s2 = batch[3]
        batch_size = sequences.shape[0]
        sequences = sequences.to(self.gluenote.device)
        targets = targets.to(self.gluenote.device)
        targets_s1 = targets_s1.to(self.gluenote.device)
        targets_s2 = targets_s2.to(self.gluenote.device)

        # compute predictions and loss
        predictions, mlp_out = self.gluenote(sequences)
        # for each column (s2) predict row (s1)
        loss_s2 = self.classification_loss(predictions[:,:,1:], targets_s2[:,1:])
        # swap and for each column (s1) predict row (s2)
        loss_s1 = self.classification_loss(torch.swapaxes(predictions,1,2)[:,:,1:], targets_s1[:,1:])

        confidence_matrix = F.softmax(predictions, 1) * F.softmax(predictions, 2)
        match_confidence = confidence_matrix[:,:,1:] # no insertions
        
        target_head = targets_s2[:,1:].detach().clone()
        output_head = self.gluehead(match_confidence)
        # output =  batch x C x seq, target = batch x seq
        loss_s3 = self.classification_loss(output_head, target_head) 
        loss = loss_s1 + loss_s2 + loss_s3
        wandb.log({"train_loss": loss.item(),
                   "s3_loss": loss_s3.item(),
                   "s2_loss": loss_s2.item(),
                   "s1_loss": loss_s1.item()})
        self.train_loss_collector.append(loss_s2.item())
        self.train_loss_collector.append(loss_s1.item())
        return loss

    def validation_step(self, batch, batch_idx):
        # unpack and move
        sequences = batch[0]
        targets = batch[1]
        targets_s1 = batch[2]
        targets_s2 = batch[3]
        sequences = sequences.to(self.gluenote.device)
        targets = targets.to(self.gluenote.device)
        targets_s1 = targets_s1.to(self.gluenote.device)
        targets_s2 = targets_s2.to(self.gluenote.device)

                # compute predictions and loss
        predictions, mlp_out = self.gluenote(sequences)
        # for each column (s2) predict row (s1)
        loss_s2 = self.classification_loss(predictions[:,:,1:], targets_s2[:,1:])
        # swap and for each column (s1) predict row (s2)
        loss_s1 = self.classification_loss(torch.swapaxes(predictions,1,2)[:,:,1:], targets_s1[:,1:])

        confidence_matrix = F.softmax(predictions, 1) * F.softmax(predictions, 2)
        match_confidence = confidence_matrix[:,:,1:] # no insertions
        
        target_head = targets_s2[:,1:].detach().clone()
        output_head = self.gluehead(match_confidence)
        # output =  batch x C x seq, target = batch x seq
        loss_s3 = self.classification_loss(output_head, target_head) 
        loss = loss_s1 + loss_s2 + loss_s3
        wandb.log({"validation_loss": loss.item()})
        self.val_collector.append((predictions, targets))
        self.val_loss_collector.append(loss_s2.item())
        self.val_loss_collector.append(loss_s1.item())
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
                    }
                }
    
    def on_validation_epoch_end(self, verbose = False):
        try:
            predictions, targets = self.val_collector[0]
            self.test_module.log_similarity_matrix(predictions[0,:,:], 
                                                    targets[0,:,:],
                                                    self.current_epoch)
            self.val_collector = list()
            accuracies = list()
            for idx in range(predictions.shape[0]):
                acc = self.test_module.compute_accuracy(predictions[0,:,:], 
                                                    targets[0,:,:],
                                                    self.current_epoch)
                accuracies.append(acc)

            print("train",np.mean(self.train_loss_collector))
            print("val",np.mean(self.val_loss_collector))
            print("accuracies",np.mean(accuracies))


            self.train_loss_collector = list()
            self.val_loss_collector = list()

        except:
            print("logging failed at epoch: ", self.current_epoch)

if __name__ == "__main__":
    pass
