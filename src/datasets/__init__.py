from torch.utils.data import Dataset, DataLoader
import miditok
from symusic import Score, Note
import glob
import numpy as np
import os
import torch

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": False,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}

class AlignmentDataset(Dataset):
    def __init__(self, 
                 dir = ".",
                 limit= False,
                 notes_per_sample = 128,
                 tokenizer = miditok.Structured(),
                 prob_insertions = 0.05,
                prob_deletions = 0.05,
                prob_repeats = 0.2,
                range_repeats = (8, 200), 
                prob_skips = 0.2,
                range_skips = (8, 200),
                # performance modifiers which affect seq
                range_tempo_curve = (0.5, 0.5), # (~N(1, x1) * 2 ** ~N(0, x2)) * tempo_before
                range_timing =  (-50, 59), # min, max of onset_before + x (in ticks)
                # other performance modifiers
                range_velocities = (-10,11),
                range_durations = (-250,250)
                ):
        self.tokenizer = tokenizer
        self.dir = dir
        self.notes_per_sample = notes_per_sample

        self.prob_insertions = prob_insertions
        self.prob_deletions = prob_deletions
        self.prob_repeats = prob_repeats
        self.range_repeats = range_repeats
        self.prob_skips = prob_skips
        self.range_skips = range_skips
        self.range_tempo_curve = range_tempo_curve
        self.range_timing =  range_timing
        self.range_velocities = range_velocities
        self.range_durations = range_durations

        print("-------")
        print("load_data")
        self.file_names = list(glob.glob(os.path.join(self.dir, "*.mid")))
        print(len(self.file_names), " files found.")
       
        # loading the data
        self.sequences = list()
        if limit:
            self.file_names = self.file_names[:5]
        self.check_files()

        print("after checking: ", len(self.file_names))
        print("-------")

    def check_files(self):
        unproblematic_files = list()
        for fn in self.file_names:
            score1 = Score(fn)
            problem = False
            if len(score1.tracks) != 1:
                problem = True
            if len(score1.tracks[0].notes) < self.notes_per_sample * 2:
                problem = True

            if problem:
                pass
                # print("PROBLEM: ", fn)

            else:
                unproblematic_files.append(fn)
        
        self.file_names = unproblematic_files

    def process_file(self, fn):
        # modify the file withing symusic and numpy
        score1 = Score(fn)
        pitches = score1.tracks[0].notes.numpy()["pitch"]
        max_pitch = pitches.max()
        min_pitch = pitches.min()
        # in tokenizer max: 'Pitch_108': 91, and min: 'Pitch_21': 4
        score1 = score1.shift_pitch(np.random.randint(min(-min_pitch + 21, 0), max(-max_pitch + 109,1)))
        score2 = score1.copy()

        t_number = 0
        score2_dict = score2.tracks[t_number].notes.numpy()
        onsets, velocities, durations, pitches, order_idx_shuffle, order_idx_og = reorder(
                score2_dict["time"],
                score2_dict["velocity"],
                score2_dict["pitch"],
                score2_dict["duration"],
                prob_insertions = self.prob_insertions,
                prob_deletions = self.prob_deletions,
                prob_repeats = self.prob_repeats,
                range_repeats = self.range_repeats, 
                prob_skips = self.prob_skips,
                range_skips = self.range_skips,
                range_tempo_curve = self.range_tempo_curve,
                range_timing = self.range_timing, 
                range_velocities = self.range_velocities,
                range_durations = self.range_durations)
        
        new_score2_dict = {
            "time": onsets,
            "velocity": velocities,
            "pitch": pitches,
            "duration": durations
        }
        
        note_list2 = Note.from_numpy(**new_score2_dict)
        score2.tracks[t_number].notes = note_list2

        # tokenizer using MIDItok
        tokens1 = self.tokenizer(score1)
        tokens2 = self.tokenizer(score2)
        # target matrix; s1 rows, s2 columns        
        target_matrix = np.zeros((len(order_idx_og), len(order_idx_shuffle)), dtype = float)
        
        # target_s1: for each row, which column is matching target
        target_s1 = np.zeros(len(order_idx_og) + 1)
        # target_s2 for each column, which row is matching target
        target_s2 = np.zeros(len(order_idx_shuffle) + 1)
        for shuffled_id, og_id in enumerate(order_idx_shuffle):
            if og_id < 1e6:
                target_matrix[og_id, shuffled_id] = 1.0
                target_s1[og_id + 1] = shuffled_id + 1
                target_s2[shuffled_id + 1] = og_id + 1
            
        if np.random.rand() > 0.5:
            sample = {
                  "s1":np.array(tokens1[t_number].ids, dtype = int),
                  "s2":np.array(tokens2[t_number].ids, dtype = int),
                  "t":target_matrix,
                  "t1":target_s1,
                  "t2":target_s2
                  }
        else: # swap the inputs
            sample = {
                  "s1":np.array(tokens2[t_number].ids, dtype = int),
                  "s2":np.array(tokens1[t_number].ids, dtype = int),
                  "t":target_matrix.T, # transpose
                  "t1":target_s2,
                  "t2":target_s1
                  }

        return sample

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return self.process_file(self.file_names[idx])
    
    def collate(self, minibatch):
        sample_list = list()
        target_list = list()
        t1_list = list()
        t2_list = list()

        for sample in minibatch:
            sample_len1 = len(sample["s1"])
            sample_len2 = len(sample["s2"])
            sample_len_min = min(sample_len1, sample_len2) // 4
            # start = 0
            start = np.random.randint(0, np.max((1,sample_len_min-self.notes_per_sample)))
            end = start + self.notes_per_sample 
            start_seq = start * 4
            end_seq = start_seq + self.notes_per_sample * 4 # for tokens per note
            s1 = sample["s1"][start_seq:end_seq]
            s2 = sample["s2"][start_seq:end_seq]
            t = sample["t"][start:end,start:end] # this probably doesn't quite fit
            t1 = sample["t1"][start+1:end+1]
            t2 = sample["t2"][start+1:end+1]
            s1,s2,t,t1,t2 = prepend_segment(s1,s2,t,t1,t2,start,start,end,end)
            sample_list.append(np.concatenate((s1,s2)).astype(int))
            target_list.append(t)
            t1_list.append(t1)
            t2_list.append(t2)
        
        return (
                torch.from_numpy(np.stack(sample_list, axis=0)).contiguous(), 
                torch.from_numpy(np.stack(target_list, axis=0)),
                torch.from_numpy(np.stack(t1_list, axis=0)).type(torch.LongTensor),
                torch.from_numpy(np.stack(t2_list, axis=0)).type(torch.LongTensor),
                )

def prepend_segment(s1,s2,
                    matrix,
                    target1,target2,
                    startidx1,startidx2,
                    endidx1,endidx2):
    new_matrix = np.zeros((matrix.shape[0]+1,matrix.shape[1]+1))
    new_matrix[1:, 1:] = matrix
    used_idx_s2 = np.sum(new_matrix, axis = 0)
    used_idx_s1 = np.sum(new_matrix, axis = 1)
    mask_s2 = used_idx_s2 == 0.0
    mask_s1 = used_idx_s1 == 0.0
    new_matrix[0, mask_s2] = 1.0
    new_matrix[mask_s1, 0] = 1.0
    # target_idx
    new_target1 = np.concatenate((np.zeros(1, dtype = int),target1))
    new_target2 = np.concatenate((np.zeros(1, dtype = int),target2))
    mask_target1 = np.any((new_target1 < startidx1 + 1,new_target1 >= endidx1 + 1), axis = 0)
    mask_target2 = np.any((new_target2 < startidx2 + 1,new_target2 >= endidx2 + 1), axis = 0)
    new_target1 += -startidx1
    new_target2 += -startidx2
    new_target1[mask_target1] = 0
    new_target2[mask_target2] = 0
    # token sequences
    new_s1 = np.concatenate((np.zeros(4, dtype = int),s1))
    new_s2 = np.concatenate((np.zeros(4, dtype = int),s2))
    return new_s1,new_s2,new_matrix,new_target1,new_target2
    
def reorder(onsets,
            velocities,
            pitches,
            durations,
            # sequence modifiers
            prob_insertions = 0.0,
            prob_deletions = 0.0,
            prob_repeats = 0.0,
            range_repeats = (4, 40), 
            prob_skips = 0.0,
            range_skips = (4, 40),
            # performance modifiers which affect seq
            range_tempo_curve = (0.0, 0.0), # (~N(1, x1) * 2 ** ~N(0, x2)) * tempo_before
            range_timing =  (0, 1), # min, max of onset_before + x (in ticks)
            # other performance modifiers
            range_velocities = (0,1),
            range_durations = (0,1)
    ):
    """
    manipulate a quartet of np arrays encoding
    onset, duration, velocity, and pitch
    to randomly add insertions, deletions,
    skip segments, and repeat segments,
    as well as add noise to all attributes.
    returns all four arrays as well as index
    array which indicates for each element where
    (if anywhere) in the old sequence it was found
    to match it to the original.

    some notes on keeping track of IDX:
    id = argsort(a) -> a[id] is sorted
    (id = np.arange(n)[id], where np.arange(n) are the initial idx)
    after that resort, and apply argsort to the initial idx
    id_1 = argsort(b)
    id = id[id_1]
    """
    
    onsets = np.copy(onsets)
    onsets_min = onsets[0]
    onsets_max = onsets[-1]
    pitches = np.copy(pitches)
    pitches_min = pitches.min()
    pitches_max = pitches.max() + 1
    durations = np.copy(durations)
    durations_min = durations.min()
    durations_max = durations.max() + 1
    velocities = np.copy(velocities)
    velocities_min = velocities.min()
    velocities_max = velocities.max() + 1

    total_num = len(onsets)
    order_idx = np.arange(len(onsets))
    order_idx_og = np.arange(len(onsets))
    
    # deletions: delete idx, onsets, keep track of idx_to_delete to delete notes
    delete_idx = np.random.choice(np.arange(total_num),size = int(prob_deletions*total_num),replace= False)
    onsets = np.delete(onsets, delete_idx)
    pitches = np.delete(pitches, delete_idx)
    velocities = np.delete(velocities, delete_idx)
    durations = np.delete(durations, delete_idx)
    order_idx = np.delete(order_idx, delete_idx)
    current_total_num = len(order_idx)

    # skips: 
    # skips, delete range of notes and adjust onsets, delete idx
    if np.random.rand() < prob_skips:
        skip_length = np.random.randint(range_skips[0], range_skips[1])
        skip_start = np.random.randint(0, current_total_num - skip_length)
        skip_delete_idx = np.arange(skip_start, skip_start+skip_length)
        onset_shift = onsets[skip_start + skip_length] - onsets[skip_start] - 23
        onsets = np.delete(onsets, skip_delete_idx)
        onsets[skip_start:] -= onset_shift
        pitches = np.delete(pitches, skip_delete_idx)
        velocities = np.delete(velocities, skip_delete_idx)
        durations = np.delete(durations, skip_delete_idx)
        order_idx = np.delete(order_idx, skip_delete_idx)
        current_total_num = len(order_idx)

    # insertions
    # random inserts, just pick random onsets and reorder, give ID of unused note
    no_of_insertions = int(prob_insertions*current_total_num)
    new_onsets = np.random.randint(onsets_min, onsets_max, no_of_insertions)
    new_velocities = np.random.randint(velocities_min, velocities_max, no_of_insertions)
    new_durations = np.random.randint(durations_min, durations_max, no_of_insertions)
    new_pitches = np.random.randint(pitches_min, pitches_max, no_of_insertions)
    # insertions marked as 1e7
    new_sorting_idx = np.full(no_of_insertions, 1e7, dtype=int)
    
    new_sorting_idx = np.concatenate((order_idx, new_sorting_idx))
    new_onsets = np.concatenate((onsets, new_onsets))
    new_velocities = np.concatenate((velocities, new_velocities))
    new_durations = np.concatenate((durations, new_durations))
    new_pitches = np.concatenate((pitches, new_pitches))
    
    sorting_idx = np.argsort(new_onsets) 
    onsets = new_onsets[sorting_idx]
    durations = new_durations[sorting_idx]
    velocities = new_velocities[sorting_idx]
    pitches = new_pitches[sorting_idx]
    order_idx = new_sorting_idx[sorting_idx]
    current_total_num = len(order_idx)


    # trills
    idx_to_trill = np.random.randint(current_total_num)
    number_of_trill_notes = np.random.randint(20,100)
    tonset = onsets[idx_to_trill]
    tvelocity = velocities[idx_to_trill]
    tduration = durations[idx_to_trill]
    tpitch = pitches[idx_to_trill]
    trill_duration = np.random.randint(10,100)
    tsecond_pitch = tpitch + np.random.randint(1,3)
    new_onsets = np.linspace(tonset,tonset + number_of_trill_notes *trill_duration,  number_of_trill_notes).astype(int)
    new_velocities = np.random.randint(velocities_min, velocities_max, number_of_trill_notes)
    new_durations = np.random.randint(trill_duration-5, trill_duration+30, number_of_trill_notes)
    new_pitches = np.zeros(number_of_trill_notes)
    new_pitches[0::2] = tsecond_pitch
    new_pitches[1::2] = tpitch
    # trills marked as insertions marked as 1e7
    new_sorting_idx = np.full(no_of_insertions, 1e7, dtype=int)
    
    new_sorting_idx = np.concatenate((order_idx, new_sorting_idx))
    new_onsets = np.concatenate((onsets, new_onsets))
    new_velocities = np.concatenate((velocities, new_velocities))
    new_durations = np.concatenate((durations, new_durations))
    new_pitches = np.concatenate((pitches, new_pitches))
    
    sorting_idx = np.argsort(new_onsets) 
    onsets = new_onsets[sorting_idx]
    durations = new_durations[sorting_idx]
    velocities = new_velocities[sorting_idx]
    pitches = new_pitches[sorting_idx]
    order_idx = new_sorting_idx[sorting_idx]
    current_total_num = len(order_idx)

    # repeats: 
    # repeats, copy range of notes and adjust onsets, give ID of unused note
    if np.random.rand() < prob_repeats:
        repeat_length = np.random.randint(range_repeats[0], range_repeats[1])
        repeat_start = np.random.randint(0, current_total_num - repeat_length)
        repeat_extraction_idx = np.arange(repeat_start, repeat_start+repeat_length)
        # insert the repeat right after
        insert_start = repeat_start + repeat_length
        onset_shift = onsets[insert_start] - onsets[repeat_start] + 23
        new_onsets = np.copy(onsets[repeat_extraction_idx])
        new_velocities = np.copy(velocities[repeat_extraction_idx])
        new_durations = np.copy(durations[repeat_extraction_idx])
        new_pitches = np.copy(pitches[repeat_extraction_idx])
        # inserted repeats marked as 2e7
        new_sorting_idx = np.full(repeat_length, 2e7, dtype = int)

        onsets = np.insert(onsets, insert_start, new_onsets)
        onsets[insert_start:] += onset_shift
        durations = np.insert(durations, insert_start, new_durations)
        velocities = np.insert(velocities, insert_start, new_velocities)
        pitches = np.insert(pitches, insert_start, new_pitches)
        order_idx = np.insert(order_idx, insert_start, new_sorting_idx)
        current_total_num = len(order_idx)

    # tempo
    tempo = np.diff(onsets)
    first_onset =  onsets[:1]
    global_tempo = np.clip(np.random.normal(1, range_tempo_curve[0]), 0.2, 5.0)
    tempo_modifier = 2**np.random.normal(0, range_tempo_curve[1], size = current_total_num-1)
    new_tempo = np.concatenate((first_onset, global_tempo * tempo * tempo_modifier))
    new_onsets = np.cumsum(new_tempo)

    # timing
    onset_timing_modifier = np.random.randint(range_timing[0], range_timing[1], current_total_num)
    new_onsets += onset_timing_modifier
    sorting_idx = np.argsort(new_onsets) 
    onsets = new_onsets[sorting_idx]
    durations = durations[sorting_idx]
    velocities = velocities[sorting_idx]
    pitches = pitches[sorting_idx]
    order_idx = order_idx[sorting_idx]
    
    # durations and velocities
    durations += np.random.randint(range_durations[0], range_durations[1], current_total_num)
    durations = np.clip(durations, 30, 500000)
    velocities += np.random.randint(range_velocities[0], range_velocities[1], current_total_num)
    velocities = np.clip(velocities, 1, 127)

    return onsets, velocities, durations, pitches, order_idx, order_idx_og


if __name__=="__main__":
    from torch.utils.data import DataLoader
    data_dir = "thegluenote/data/nasap"
    full_dataset = AlignmentDataset(data_dir,
                                    notes_per_sample = 16)    

    data_loader = DataLoader(dataset=full_dataset, batch_size = 1, collate_fn=full_dataset.collate)

    # Using the data loader in the training loop
    for batch in data_loader:
        
        print("Train your model on this batch...")
        break
    