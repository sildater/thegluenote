CONFIG_DEFAULTS = dict(
    dropout=0.2,
    transformer_dim=16,
    sequence_len=16,
    transformer_blocks=2,
    num_heads = 8,
    batch_size=4,
    learning_rate=5e-4,
    epochs=5,
    prop_del = 0.0,
    prob_ins = 0.0,
    prob_rep = 0.0,
    prob_ski = 0.0)

CONFIG_DEFAULTS_LARGE = dict( 
    dropout=0.2,
    transformer_dim=512,
    sequence_len=512,
    transformer_blocks=8,
    decoder_blocks = 8
    num_heads = 8,
    batch_size = 4,
    learning_rate = 3e-4, 
    epochs = 400,
    prob_del = 0.2,
    prob_ins = 0.2,
    prob_rep = 1.0,
    prob_ski = 1.0)

CONFIG_DEFAULTS_MID = dict( 
    dropout=0.2,
    transformer_dim = 256,
    sequence_len = 512,
    transformer_blocks = 6,
    decoder_blocks = 2,
    num_heads = 8,
    batch_size= 16,
    learning_rate = 5e-4,
    epochs = 1200, # 3200
    prob_del = 0.2,
    prob_ins = 0.2,
    prob_rep = 1.0,
    prob_ski = 1.0)

CONFIG_DEFAULTS_SMALL = dict( 
    dropout=0.2,
    transformer_dim=128,
    sequence_len=512,
    transformer_blocks=4,
    decoder_blocks=2,
    num_heads = 8,
    batch_size= 24,
    learning_rate=5e-4,
    epochs=3000, # 4800
    prob_del = 0.2,
    prob_ins = 0.2,
    prob_rep = 1.0,
    prob_ski = 1.0)

class Config:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)

