import miditok
from symusic import Score, Note
import numpy as np
import glob
import os
import torch.nn.functional as F
import time
import parangonar as pa
from parangonar.match.utils import expand_grace_notes #,convert_grace_to_insertions 
import wandb
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import partitura as pt
import torch
import json

from post import (get_local_path_from_confidence_matrix,
                get_input_to_ref_map,                
                get_note_matches_with_updating_map)

DEFAULT_NOTE = [
                196, # 'TimeShift_1.0.8'
                4, # 'Pitch_21'
                107, # 'Velocity_63'
                131 # 'Duration_1.0.8'
                ]

class TestAlignmentModule:
    def __init__(self, 
                 dir = ".",
                 file_set = "4x22",
                 tokenizer = miditok.Structured(),
                 log_to_wandb = False,
                 unmatched_idx = 100000000,
                 matching_threshold = 0.5,
                 max_calls = 2,
                ):
        self.dir = dir
        self.file_set = file_set
        self.log_to_wandb = log_to_wandb
        self.tokenizer = tokenizer
        self.matching_threshold = matching_threshold
        self.unmatched_idx = unmatched_idx
        self.max_calls = max_calls
        self.test_files = list()
        
        for dump_path in glob.glob(os.path.join(self.dir, self.file_set,"*.npz")):
            bn = os.path.basename(dump_path)
            npzfile = np.load(dump_path, allow_pickle=True)
            keys = [k for k in npzfile.keys()]
            if 'gt_alignment_insertions' in keys:
                gt_alignment = list(npzfile['gt_alignment_insertions'])
            else:
                gt_alignment = list(npzfile['alignment'])
            if 'score_note_array_full' in keys:
                score_note_array_full = npzfile['score_note_array_full']
            else:
                score_note_array_full = npzfile['score_note_array']
            performance_note_array = npzfile['performance_note_array']

            (performance_note_array,
            score_note_array_full) = clean_note_arrays_for_alignment(performance_note_array,
                                                                        score_note_array_full,
                                                                        gt_alignment)

            score_note_array_full = score_note_array_full[score_note_array_full["onset_beat"].argsort()]
            path_empty_midi = os.path.join(self.dir, "empty480ppq.mid")
            performance_midi = note_array_to_symusic_score(performance_note_array, path_empty_midi)
            score_midi = note_array_to_symusic_score(score_note_array_full, path_empty_midi)

            self.test_files.append( {
                "performance":performance_note_array,
                "score":score_note_array_full,
                "gt_alignment":gt_alignment,
                "alignment":gt_alignment,
                "piece_name":bn.split(".")[0],
                "score_midi": score_midi,
                "performance_midi": performance_midi
            })
    
    def log_similarity_matrix(self, 
                              predictions, 
                              targets,
                              epoch=None):
        # figure of similarity matrices: predicted and target
        # log to: wandb or locally as png
        if epoch is None:
            epoch = 0
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        fig,a = plt.subplots(2, 1, figsize=(6, 6))
        ax = a[0].imshow(predictions)
        a[1].imshow(targets)
        fig.colorbar(ax, ax=a, orientation='vertical', fraction=0.1)
        ACC = (np.argmax(predictions[1:, 0:], axis = 1) == np.argmax(targets[1:, 0:], axis = 1)).sum() / (targets.shape[0] - 1)
        if self.log_to_wandb:
            wandb.log({"pred targ at epoch "+str(epoch)+ " acc "+str(ACC): fig})
        else:
            figs_dir = os.path.join(self.dir, "figs")
            if not os.path.isdir(figs_dir):
                os.makedirs(figs_dir)
            plt.savefig(os.path.join(figs_dir, "pred_targ_at_epoch_"+str(epoch)+ "_acc_"+str(ACC)+".png"))
        plt.close(fig)

    def compute_accuracy(self, 
                        predictions, 
                        targets,
                        epoch=None):
        if epoch is None:
            epoch = 0
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        ACC = (np.argmax(predictions[1:, 0:], axis = 1) == np.argmax(targets[1:, 0:], axis = 1)).sum() / (targets.shape[0] - 1)
        return ACC

    def save_results_model_tester(self, 
                                epoch = None,
                                log_parangonada = False,
                                log_figure_wandb = False,
                                log_json = False):
        # Table of alignment results: wandb or print
        # figure of note alignments: wandb
        # Export to parangonada files: local
        # Export to json: local

        if epoch is None:
            epoch = 0
        if self.log_to_wandb:
            table = wandb.Table(columns=["Piece", "Type", "Precision_", "Recall_", "F-Score_"])

        for elem in self.test_files:
            if not self.log_to_wandb:
                print("*"*100)
                print( elem["piece_name"])
            performance = elem["performance"]
            score = elem["score"]
            gt_alignment = elem["gt_alignment"]
            alignment = elem["alignment"]
            piece_name = elem["piece_name"]
            types = ['match','insertion', 'deletion']
            local_info = elem["local_info"]
            for t in types:
                precision, recall, f_score, len_p, len_gt = pa.fscore_alignments(alignment, 
                                                                                gt_alignment, 
                                                                                t,
                                                                                return_numbers = True)
                
                if self.log_to_wandb:
                    table.add_data(piece_name, t, 
                                   format(precision, '.3f'), format(recall, '.3f'),format(f_score, '.3f'))

                else: 
                    if t == "match":
                        print("*"*50)
                        print(piece_name, "type:  ", t )
                        # print('------------------')
                        # print("alignment gt len: ", len(gt_alignment))
                        # print("alignment len: ", len(alignment))
                        print('------------------: ' )
                        print('Precision: ',format(precision, '.3f'),
                            'Recall ',format(recall, '.3f'),
                            'F-Score ',format(f_score, '.3f'))
                        print('------------------')
                        # print('p: ',len_p,
                        #     'gt: ',len_gt)
                        
                if log_json:
                    local_info['Precision_'+t] = format(precision, '.3f')
                    local_info['Recall_'+t] = format(recall, '.3f')
                    local_info['F-Score_'+t] = format(f_score, '.3f')

            
            fields = ["onset_beat","duration_beat",
                      "onset_quarter","duration_quarter",
                      "onset_div","duration_div",
                      "pitch","voice","id"]
            score_note_array_full=expand_grace_notes(score)[fields]
            if self.log_to_wandb and log_figure_wandb:
                # plot note alignment
                fig = pa.evaluate.plot_alignment_comparison(performance, 
                                            score_note_array_full, 
                                            alignment,
                                            gt_alignment,
                                            return_figure = True,
                                            figsize = (40, 5))
                wandb.log({"plot "+piece_name+ " at epoch: "+ str(epoch): fig})
                plt.close(fig)

            if log_parangonada:
                # save parangonada data
                parangonada_dir = os.path.join(self.dir, "parangonada")
                if not os.path.isdir(parangonada_dir):
                    os.makedirs(parangonada_dir)
                piece_parangonada_dir = os.path.join(parangonada_dir, piece_name+ "_epoch_"+ str(epoch)) 
                if not os.path.isdir(piece_parangonada_dir):
                    os.makedirs(piece_parangonada_dir)  
                pa.match.save_parangonada_csv(alignment, 
                                                performance,
                                                score_note_array_full,
                                                outdir=piece_parangonada_dir,
                                                zalign=gt_alignment)

        if self.log_to_wandb:
            wandb.log({"Note Alignment Quality at epoch: "+ str(epoch): table})

        if log_json:
            info = dict()
            for elem in self.test_files:
                piece_name = elem["piece_name"]
                local_info = elem["local_info"]
                info[piece_name] = local_info

            with open(str(log_json)+".json", "w") as fp: 
                json.dump(info, fp) 

    def model_tester(self, 
                     model, 
                     decoder_model = None,
                     matching_type = "head",
                     epoch = None,
                     log_parangonada = False,
                     log_json = False):
        model.eval()
        if decoder_model is not None:
            decoder_model.eval()

        for idx, elem in enumerate(self.test_files): 
            t1 = time.time()
            local_info = dict()
            performance_note_array = elem["performance"]
            score_note_array_full = elem["score"]
            performance_midi = elem["performance_midi"]
            score_midi = elem["score_midi"]  
            gt_alignment = elem["gt_alignment"]  
            piece_name = elem["piece_name"]
            print("testing:", piece_name, len(score_note_array_full), len(performance_note_array))
            formatted_gt_alignment_idx = format_note_array_alignment(score_note_array_full,
                                                                     performance_note_array,
                                                                     gt_alignment,
                                                                     unmatched_idx = self.unmatched_idx)
            t2 = time.time()

            if matching_type == "head":
                alignment =    get_predictions_from_head_model(
                                                                encoder_model = model, 
                                                                head_model = decoder_model,
                                                                tokenizer = self.tokenizer,
                                                                input_midi1 = score_midi, 
                                                                input_midi2 = performance_midi,
                                                                unmatched_idx = self.unmatched_idx)
            
            elif matching_type == "matrix":
                alignment = get_matrix_alignment_from_model(encoder_model = model, 
                                        tokenizer = self.tokenizer,
                                        input_midi1 = score_midi, 
                                        input_midi2 = performance_midi,
                                        unmatched_idx = self.unmatched_idx, 
                                        formatted_gt_alignment_idx = formatted_gt_alignment_idx,
                                        piece_name = elem["piece_name"])


            elif matching_type == "dtw":
                alignment = get_dtw_alignment_from_model(encoder_model = model, 
                                        tokenizer = self.tokenizer,
                                        input_midi1 = score_midi, 
                                        input_midi2 = performance_midi,
                                        unmatched_idx = self.unmatched_idx, 
                                        formatted_gt_alignment_idx = formatted_gt_alignment_idx,
                                        piece_name = elem["piece_name"])
            
            t3 = time.time()
            formatted_alignment = format_score_performance_alignment(score_note_array_full,
                                                                     performance_note_array,
                                                                     alignment,
                                                                     unmatched_idx = self.unmatched_idx)
            

            self.test_files[idx]["alignment"] = formatted_alignment
            t4 = time.time()
            local_info["preprocessing_time"]= "{:3.2f}".format(t2-t1),
            local_info["alignment_time"]= "{:3.2f}".format(t3-t2),
            local_info["eval_time"]=  "{:3.2f}".format(t4-t3)
            self.test_files[idx]["local_info"] = local_info

        self.save_results_model_tester(epoch = epoch, 
                                       log_parangonada = log_parangonada,
                                       log_json = log_json)

# prediction functions

def get_predictions_from_head_model(encoder_model, 
                                       head_model,
                                        tokenizer,
                                        input_midi1, 
                                        input_midi2,
                                        unmatched_idx = 100000000):
    
    sequence_length = head_model.position_number
    input_midi2, input_midi1 = get_shifted_and_stretched_and_agnostic_midis(input_midi2, input_midi1)
    tokens1 = tokenizer(input_midi1)
    tokens2 = tokenizer(input_midi2)
    sample = {"s1":np.array(tokens1[0].ids, dtype = int),
              "s2":np.array(tokens2[0].ids, dtype = int)}
    max_len1 = len(input_midi1.tracks[0].notes) 
    max_len2 = len(input_midi2.tracks[0].notes)
    current_idx_1= [0, sequence_length]
    current_idx_2= [0, sequence_length]
    running = True
    alignment_collector = list()
    match_collector = list()
    used_x1 = list()
    while running:
        # pad and prepare the note sequences
        s1 = sample["s1"][current_idx_1[0]*4:current_idx_1[1]*4]
        s2 = sample["s2"][current_idx_2[0]*4:current_idx_2[1]*4]
        s1_matrix_end = sequence_length
        s2_matrix_end = sequence_length
        padding_len1 = 0
        padding_len2 = 0
        if len(s1) < sequence_length * 4:
            # pad the sequence
            padding_len1 = sequence_length * 4 - len(s1)
            padding1 = np.array(DEFAULT_NOTE * int(padding_len1 / 4))
            s1_matrix_end = int(len(s1) / 4)
            s1 = np.concatenate((s1,padding1)).astype(int)
        if len(s2) < sequence_length * 4:
            # pad the sequence
            padding_len2 = sequence_length * 4 - len(s2)
            padding2 = np.array(DEFAULT_NOTE * int(padding_len2 / 4))
            s2_matrix_end = int(len(s2) / 4)
            s2 = np.concatenate((s2,padding2)).astype(int)
        if padding_len2 > 0 or padding_len1 > 0:
            running = False
        s1 = np.concatenate((np.zeros(4), s1)).astype(int)
        s2 = np.concatenate((np.zeros(4), s2)).astype(int)
        sequences = torch.from_numpy(np.concatenate((s1,s2)).astype(int)).contiguous().unsqueeze(0)
        
        # call the encoder
        sequences = sequences.to(encoder_model.device)
        predictions, mlp_out = encoder_model(sequences)
        confidence_matrix = F.softmax(predictions, 1) * F.softmax(predictions, 2)
        match_confidence = confidence_matrix[:,:,1:] # no insertions
        # batch x C x seq
        output_head = head_model(match_confidence)
        for x2 in range(sequence_length): 
            
            x1_max = torch.argmax(output_head[0,:,x2]).item()
            
            if x1_max > 0 and current_idx_1[0] + x1_max - 1 < max_len1 and current_idx_2[0] + x2 < max_len2:
                alignment_collector.append([current_idx_1[0] + x1_max - 1, current_idx_2[0] +  x2 ]) # sid, pid
                used_x1.append(current_idx_1[0] + x1_max - 1)
            elif  x1_max == 0 and current_idx_2[0] + x2 < max_len2: 
                alignment_collector.append([unmatched_idx, current_idx_2[0] + x2 ])

        current_idx_1[0] += 512
        current_idx_1[1] += 512
        current_idx_2[0] += 512
        current_idx_2[1] += 512

    for x1 in range(0, max_len1):
            if x1 not in used_x1:
                alignment_collector.append([ x1, unmatched_idx ])
    alignment = np.array(alignment_collector)
    return alignment

def get_dtw_alignment_from_model(encoder_model, 
                            tokenizer,
                            input_midi1, 
                            input_midi2,
                            unmatched_idx = 100000000, 
                            formatted_gt_alignment_idx = None,
                            piece_name = ""):
    # setup and preprocessing of files
    sequence_length = encoder_model.position_number - 1
    input_midi2, input_midi1 = get_shifted_and_stretched_and_agnostic_midis(input_midi2, input_midi1)
    note_array = minimal_note_array_from_symusic(input_midi1)
    note_array_ref = minimal_note_array_from_symusic(input_midi2)
    tokens1 = tokenizer(input_midi1)
    tokens2 = tokenizer(input_midi2)
    sample = {"s1":np.array(tokens1[0].ids, dtype = int),
              "s2":np.array(tokens2[0].ids, dtype = int)}
    
    no_notes_s1 = len(note_array)
    no_notes_s2 = len(note_array_ref)
    index_shift = int(sequence_length/2)
    no_slices_s1 = no_notes_s1 // (index_shift) + 1
    no_slices_s2 = no_notes_s2 // (index_shift) + 1
    full_similarity_matrix = np.zeros((no_notes_s1, no_notes_s2))

    # loop over windows
    for i in range(no_slices_s1):
        for j in range(no_slices_s2):
            current_idx_1 = [i*index_shift, (i+2)*index_shift]
            current_idx_2 = [j*index_shift, (j+2)*index_shift]

            # pad and prepare the note sequences
            s1 = sample["s1"][current_idx_1[0]*4:current_idx_1[1]*4]
            s2 = sample["s2"][current_idx_2[0]*4:current_idx_2[1]*4]
            s1_matrix_end = sequence_length
            s2_matrix_end = sequence_length

            if len(s1) < sequence_length * 4:
                # pad the sequence
                padding_len1 = sequence_length * 4 - len(s1)
                padding1 = np.array(DEFAULT_NOTE * int(padding_len1 / 4))#np.zeros(padding_len1)
                s1_matrix_end = int(len(s1) / 4)
                s1 = np.concatenate((s1,padding1)).astype(int)

            if len(s2) < sequence_length * 4:
                # pad the sequence
                padding_len2 = sequence_length * 4 - len(s2)
                padding2 = np.array(DEFAULT_NOTE * int(padding_len2 / 4))#np.zeros(padding_len2)
                s2_matrix_end = int(len(s2) / 4)
                s2 = np.concatenate((s2,padding2)).astype(int)

            s1 = np.concatenate((np.zeros(4), s1)).astype(int)
            s2 = np.concatenate((np.zeros(4), s2)).astype(int)
            sequences = torch.from_numpy(np.concatenate((s1,s2)).astype(int)).contiguous().unsqueeze(0)
            
            # call the encoder
            sequences = sequences.to(encoder_model.device)
            predictions, _ = encoder_model(sequences)
            confidence_matrix = F.softmax(predictions, 1) * F.softmax(predictions, 2)
            confidence_matrix_segment = confidence_matrix[0,1:s1_matrix_end + 1,1:s2_matrix_end + 1].detach().cpu().numpy()
            full_similarity_matrix[current_idx_1[0]:current_idx_1[1],current_idx_2[0]:current_idx_2[1]] += confidence_matrix_segment
            
    now = time.time()
    (path, 
    starting_path, ending_path, 
    s1_exclusion_start, s1_exclusion_end) = get_local_path_from_confidence_matrix(full_similarity_matrix)
    then = time.time()
    print("DTW local path time: ", then - now)
    print("PATH length:", len(path), full_similarity_matrix.shape)
        
    s1_to_s2_map = get_input_to_ref_map(note_array,
                                        note_array_ref,
                                        path,
                                        return_callable = False)
    
    alignment = get_note_matches_with_updating_map(note_array,
                                    note_array_ref,
                                    s1_to_s2_map,
                                    onset_threshold = 1000,
                                    unmatched_idx = unmatched_idx)
    
    return alignment

def get_matrix_alignment_from_model(encoder_model, 
                            tokenizer,
                            input_midi1, 
                            input_midi2,
                            unmatched_idx = 100000000, 
                            formatted_gt_alignment_idx = None,
                            piece_name = ""):
    # setup and preprocessing of files
    sequence_length = encoder_model.position_number - 1
    input_midi2, input_midi1 = get_shifted_and_stretched_and_agnostic_midis(input_midi2, input_midi1)
    note_array = minimal_note_array_from_symusic(input_midi1)
    note_array_ref = minimal_note_array_from_symusic(input_midi2)
    max_len1 = len(note_array)
    max_len2 = len(note_array_ref)
    tokens1 = tokenizer(input_midi1)
    tokens2 = tokenizer(input_midi2)
    sample = {"s1":np.array(tokens1[0].ids, dtype = int),
              "s2":np.array(tokens2[0].ids, dtype = int)}
    
    no_notes_s1 = len(note_array)
    no_notes_s2 = len(note_array_ref)
    index_shift = int(sequence_length/2)
    no_slices_s1 = no_notes_s1 // (index_shift) + 1
    no_slices_s2 = no_notes_s2 // (index_shift) + 1
    full_similarity_matrix = np.zeros((no_notes_s1 + 1, no_notes_s2))

    # loop over windows
    for i in range(no_slices_s1):
        for j in range(no_slices_s2):
            current_idx_1 = [i*index_shift, (i+2)*index_shift]
            current_idx_2 = [j*index_shift, (j+2)*index_shift]

            # pad and prepare the note sequences
            s1 = sample["s1"][current_idx_1[0]*4:current_idx_1[1]*4]
            s2 = sample["s2"][current_idx_2[0]*4:current_idx_2[1]*4]
            s1_matrix_end = sequence_length
            s2_matrix_end = sequence_length

            if len(s1) < sequence_length * 4:
                # pad the sequence
                padding_len1 = sequence_length * 4 - len(s1)
                padding1 = np.array(DEFAULT_NOTE * int(padding_len1 / 4))#np.zeros(padding_len1)
                s1_matrix_end = int(len(s1) / 4)
                s1 = np.concatenate((s1,padding1)).astype(int)

            if len(s2) < sequence_length * 4:
                # pad the sequence
                padding_len2 = sequence_length * 4 - len(s2)
                padding2 = np.array(DEFAULT_NOTE * int(padding_len2 / 4))#np.zeros(padding_len2)
                s2_matrix_end = int(len(s2) / 4)
                s2 = np.concatenate((s2,padding2)).astype(int)

            s1 = np.concatenate((np.zeros(4), s1)).astype(int)
            s2 = np.concatenate((np.zeros(4), s2)).astype(int)
            sequences = torch.from_numpy(np.concatenate((s1,s2)).astype(int)).contiguous().unsqueeze(0)
            
            # call the encoder
            sequences = sequences.to(encoder_model.device)
            predictions, _ = encoder_model(sequences)
            confidence_matrix = F.softmax(predictions, 1) * F.softmax(predictions, 2)
            confidence_matrix_segment = confidence_matrix[0,1:s1_matrix_end + 1,1:s2_matrix_end + 1].detach().cpu().numpy()

            full_similarity_matrix[current_idx_1[0] + 1 :current_idx_1[1] + 1,current_idx_2[0]:current_idx_2[1]] += confidence_matrix_segment
            confidence_matrix_unmatched = confidence_matrix[0,0,1:s2_matrix_end + 1].detach().cpu().numpy()
            full_similarity_matrix[0,current_idx_2[0]:current_idx_2[1]] += confidence_matrix_unmatched

    alignment_collector = list()
    used_x1 = list()
    for x2 in range(full_similarity_matrix.shape[1]): 
        x1_max = np.argmax(full_similarity_matrix[:,x2])
        if x1_max > 0 and x1_max - 1 < max_len1 and x2 < max_len2:
            alignment_collector.append([x1_max - 1, x2 ]) # sid, pid
            used_x1.append(x1_max - 1)
        elif  x1_max == 0 and x2 < max_len2: 
            alignment_collector.append([unmatched_idx, x2 ])

    for x1 in range(0, max_len1):
        if x1 not in used_x1:
            alignment_collector.append([ x1, unmatched_idx ])
    alignment = np.array(alignment_collector)
    return alignment

# other utilities
    
# ----------------- symusic score <-> note array

def note_array_to_symusic_score(note_array, 
                                path_empty_midi):
    fields = set(note_array.dtype.fields)
    score_units = set(("onset_quarter", "onset_beat"))
    performance_units = set(("onset_tick", "onset_sec"))

    if len(score_units.intersection(fields)) > 0: 
        onset_field =  "onset_quarter"
        duration_field = "duration_quarter"
        time_conversion = 480
    elif len(performance_units.intersection(fields)) > 0:
        if len(set(("onset_tick")).intersection(fields)) > 0:
            onset_field =  "onset_tick"
            duration_field = "duration_tick"
            time_conversion = 1
        else:
            onset_field =  "onset_sec"
            duration_field = "duration_sec"
            time_conversion = 480

    # this is an ugly hack because I don't know how to create a track populated empty score object
    symusic_container = Score(path_empty_midi)
    time = note_array[onset_field] * time_conversion
    duration = note_array[duration_field] * time_conversion
    pitch = note_array["pitch"]
    if "velocity" in fields:
        velocity = note_array["velocity"] 
    else:
        velocity = np.full_like(note_array[onset_field] , 64)

    note_info =     {'time': time.astype(np.int32),
                    'duration': duration.astype(np.int32),
                    'pitch': pitch.astype(np.int8),
                    'velocity': velocity.astype(np.int8)}

    symusic_note_list = Note.from_numpy(**note_info)
    symusic_container.tracks[0].notes = symusic_note_list
    return symusic_container

def minimal_note_array_from_symusic(score,
                                    fields = ["pitch", "time"]):
    note_info = score.tracks[0].notes.numpy()
    note_array = np.column_stack([note_info[field] for field in fields])
    return note_array

# ----------------- preprocessing symusic scores

def get_shifted_and_stretched_and_agnostic_midis(midi1, midi2):
    note_info2 =midi2.tracks[0].notes.numpy()
    note_info1 =midi1.tracks[0].notes.numpy()
    # shift to zero
    note_info2["time"] -= note_info2["time"].min()
    # stretch to the same length
    note_info1 = stretch(note_info1, note_info2, factor = 1.0)
    note_info1 = velocity_and_duration_agnostic_note_info(note_info1)
    note_info2 = velocity_and_duration_agnostic_note_info(note_info2)
    symusic_note_list1 = Note.from_numpy(**note_info1)
    symusic_note_list2 = Note.from_numpy(**note_info2)
    midi1.tracks[0].notes = symusic_note_list1
    midi2.tracks[0].notes = symusic_note_list2
    return midi1, midi2

def velocity_and_duration_agnostic_note_info(note_info):
    new_duration = np.full_like(note_info["duration"], 100)
    new_velocity = np.full_like(note_info["velocity"], 63)
    note_info["velocity"] = new_velocity
    note_info["duration"] = new_duration
    return note_info

def stretch(note_info_to_change, note_info_ref, factor = 2.0):
    time_to_change = note_info_to_change["time"]
    dur_to_change = note_info_to_change["duration"]
    time_ref = note_info_ref["time"]

    min_to_change = time_to_change.min()
    max_to_change = time_to_change.max()
    min_ref = time_ref.min()
    max_ref = time_ref.max()

    stretch_factor = (max_ref - min_ref) * factor / (max_to_change - min_to_change)
    new_time = (time_to_change - min_to_change) * stretch_factor
    new_dur = dur_to_change * stretch_factor

    return  {'time': new_time.astype(np.int32),
            'duration': new_dur.astype(np.int32),
            'pitch': note_info_to_change["pitch"].astype(np.int8),
            'velocity': note_info_to_change["velocity"].astype(np.int8)}

# ----------------- alignment by note names to/from alignment by index 

def format_score_performance_alignment(score_note_array, 
                                       performance_note_array, 
                                       alignment_idx, 
                                       unmatched_idx):          
    alignment = list()
    for sidx, pidx in alignment_idx:
        if sidx < unmatched_idx and pidx < unmatched_idx:
            alignment.append({'label': 'match', 'score_id': score_note_array["id"][sidx], 'performance_id': performance_note_array["id"][pidx]})    
        else:
            if sidx < unmatched_idx:
                alignment.append({'label': 'deletion', 'score_id': score_note_array["id"][sidx]})
            if pidx < unmatched_idx:
                alignment.append({'label': 'insertion', 'performance_id': performance_note_array["id"][pidx]})

    return alignment

def format_note_array_alignment(score_note_array, 
                                performance_note_array, 
                                alignment, 
                                unmatched_idx):          
    alignment_idx = list()
    score_note_name_to_index = {sid:idx for idx, sid in enumerate(score_note_array["id"])}
    performance_note_name_to_index = {pid:idx for idx, pid in enumerate(performance_note_array["id"])}
    for match in alignment:
        if match["label"] == "match":
            alignment_idx.append([score_note_name_to_index[match['score_id']], performance_note_name_to_index[match["performance_id"]]])
        elif match["label"] == "deletion":
            alignment_idx.append([score_note_name_to_index[match['score_id']], unmatched_idx])
        elif match["label"] == "insertion":
            alignment_idx.append([unmatched_idx, performance_note_name_to_index[match["performance_id"]]])
    return alignment_idx

def clean_note_arrays_for_alignment(p_note_array,
                                    s_note_array,
                                    gt_alignment):
    used_sids = list()
    used_pids = list()
    for match in gt_alignment:

        if match["label"] == "match":
            used_sids.append(match['score_id'])
            used_pids.append(match["performance_id"])
        elif match["label"] == "deletion":
            used_sids.append(match['score_id'])
        elif match["label"] == "insertion":
            used_pids.append(match["performance_id"])

    s_mask = list()
    for sid in s_note_array["id"]:
        if sid in used_sids:
            s_mask.append(True)
        else:
            s_mask.append(False)

    p_mask = list()
    for pid in p_note_array["id"]:
        if pid in used_pids:
            p_mask.append(True)
        else:
            p_mask.append(False)

    new_s_note_array = s_note_array[np.array(s_mask)]
    new_p_note_array = p_note_array[np.array(p_mask)]
    return new_p_note_array, new_s_note_array

if __name__ == "__main__":
    pass
