
from parangonar.match.dtw import WDTW, invert_matrix, FDTW
from parangonar.match.matchers import unique_alignments
import numpy as np
import os
from scipy.interpolate import interp1d
from collections import defaultdict
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# utilities

def insert_matches_into_matched_seqs(matched_onset_seqs,
                                     new_matches):
    new_matched_onset_seqs = np.copy(matched_onset_seqs)

    new_lines = list()
    idx_to_delete = list()
    for match in new_matches:
        id_seq1 = np.where(matched_onset_seqs[:,0] == match[0])[0]
        if len(id_seq1) > 0:
           idx_to_delete.append(id_seq1[0])
           new_lines.append(match)

    if len(new_lines) > 0:
        new_lines_numpy = np.array(new_lines)
        deletion_mask = np.array(idx_to_delete)
        new_matched_onset_seqs = np.delete(new_matched_onset_seqs, deletion_mask, axis=0)
        new_matched_onset_seqs = np.concatenate((new_matched_onset_seqs, new_lines_numpy))
        new_matched_onset_seqs = new_matched_onset_seqs[np.argsort(new_matched_onset_seqs[:,0])]

    return new_matched_onset_seqs

def get_note_matches_with_updating_map(note_array,# pitch, onset
                     note_array_ref,# pitch, onset
                     matched_onset_seqs,
                     onset_threshold,
                     unmatched_idx = 100000000):
     
    note_array_idx_range = np.arange(len(note_array))
    note_array_ref_idx_range = np.arange(len(note_array_ref))

    # Get symbolic note_alignments
    note_alignments = list()
    used_note_ids = set()
    used_ref_note_ids = set()
    unique_pitches, pitch_counts = np.unique(np.concatenate((note_array[:,0],note_array_ref[:,0]), axis= 0), return_counts = True)
    pitch_by_quantity = np.argsort(pitch_counts)

    for pitch in unique_pitches[pitch_by_quantity]:

        input_to_ref_map = interp1d(matched_onset_seqs[:,0],
                                    matched_onset_seqs[:,1],
                                    kind = "linear",
                                    fill_value = "extrapolate")

        note_array_pitch_mask = note_array[:,0] == pitch
        note_array_ref_pitch_mask = note_array_ref[:,0] == pitch
        
        note_array_onsets = note_array[note_array_pitch_mask,1]
        note_array_ref_onsets = note_array_ref[note_array_ref_pitch_mask,1]

        note_array_ids = note_array_idx_range[note_array_pitch_mask]
        note_array_ref_ids = note_array_ref_idx_range[note_array_ref_pitch_mask]
        
        estimated_note_array_ref_onsets = input_to_ref_map(note_array_onsets)

        

        if  (note_array_ref_onsets.shape[0] > 1 and note_array_onsets.shape[0] > 1) or \
            (note_array_ref_onsets.shape[0] > 1 and note_array_onsets.shape[0] == 1) or \
            (note_array_ref_onsets.shape[0] == 1 and note_array_onsets.shape[0] > 1):
            try:
                ID_tuples = unique_alignments(estimated_note_array_ref_onsets, 
                                              note_array_ref_onsets,
                                              threshold=onset_threshold)
            except:
                import pdb; pdb.set_trace()
        
        elif note_array_ref_onsets.shape[0] == 1 and note_array_onsets.shape[0] == 1: 
            if np.abs(estimated_note_array_ref_onsets[0] - note_array_ref_onsets[0]) < onset_threshold:
                ID_tuples = [(0,0)]
            else:
                ID_tuples = []
        else:
            ID_tuples = []       
        
        for input_idx, ref_idx in ID_tuples:
            note_alignments.append(
                [note_array_ids[input_idx], note_array_ref_ids[ref_idx]]

            )
            used_note_ids.add(note_array_ids[input_idx])
            used_ref_note_ids.add(note_array_ref_ids[ref_idx])
        
        if len(ID_tuples) > 0:
            ID_tuples_numpy = np.array(ID_tuples)
            new_matches = np.column_stack((note_array_onsets[ID_tuples_numpy[:,0]],note_array_ref_onsets[ID_tuples_numpy[:,1]]))
            matched_onset_seqs = insert_matches_into_matched_seqs(matched_onset_seqs,
                                                                new_matches)


    # add unmatched notes
    for note_idx in note_array_idx_range:
        if note_idx not in used_note_ids:
            note_alignments.append([note_idx, unmatched_idx])

    for ref_idx in note_array_ref_idx_range:
        if ref_idx not in used_ref_note_ids:
            note_alignments.append([unmatched_idx, ref_idx])
   

    note_alignments = np.array(note_alignments)
    note_alignments = note_alignments[np.argsort(note_alignments[:,0]),:]
    return note_alignments

def get_note_matches(note_array,# pitch, onset
                     note_array_ref,# pitch, onset
                     input_to_ref_map,
                     onset_threshold,
                     unmatched_idx = 100000000):
     
    note_array_idx_range = np.arange(len(note_array))
    note_array_ref_idx_range = np.arange(len(note_array_ref))

    # Get symbolic note_alignments
    note_alignments = list()
    used_note_ids = set()
    used_ref_note_ids = set()
    unique_pitches = np.unique(np.concatenate((note_array[:,0],note_array_ref[:,0]), axis= 0))

    for pitch in unique_pitches:

        note_array_pitch_mask = note_array[:,0] == pitch
        note_array_ref_pitch_mask = note_array_ref[:,0] == pitch
        
        note_array_onsets = note_array[note_array_pitch_mask,1]
        note_array_ref_onsets = note_array_ref[note_array_ref_pitch_mask,1]

        note_array_ids = note_array_idx_range[note_array_pitch_mask]
        note_array_ref_ids = note_array_ref_idx_range[note_array_ref_pitch_mask]
        
        estimated_note_array_ref_onsets = input_to_ref_map(note_array_onsets)

        

        if  (note_array_ref_onsets.shape[0] > 1 and note_array_onsets.shape[0] > 1) or \
            (note_array_ref_onsets.shape[0] > 1 and note_array_onsets.shape[0] == 1) or \
            (note_array_ref_onsets.shape[0] == 1 and note_array_onsets.shape[0] > 1):
            try:
                ID_tuples = unique_alignments(estimated_note_array_ref_onsets, 
                                              note_array_ref_onsets,
                                              threshold=onset_threshold)
            except:
                import pdb; pdb.set_trace()
        
        elif note_array_ref_onsets.shape[0] == 1 and note_array_onsets.shape[0] == 1: 
            if np.abs(estimated_note_array_ref_onsets[0] - note_array_ref_onsets[0]) < onset_threshold:
                ID_tuples = [(0,0)]
            else:
                ID_tuples = []
        else:
            ID_tuples = []       
        
        for input_idx, ref_idx in ID_tuples:
            note_alignments.append(
                [note_array_ids[input_idx], note_array_ref_ids[ref_idx]]

            )
            used_note_ids.add(note_array_ids[input_idx])
            used_ref_note_ids.add(note_array_ref_ids[ref_idx])

        if pitch == 96:
            print("score", note_array_onsets)
            print("proj", estimated_note_array_ref_onsets)
            print("perf", note_array_ref_onsets)
            print("alignment", ID_tuples)

    # add unmatched notes
    for note_idx in note_array_idx_range:
        if note_idx not in used_note_ids:
            note_alignments.append([note_idx, unmatched_idx])

    for ref_idx in note_array_ref_idx_range:
        if ref_idx not in used_ref_note_ids:
            note_alignments.append([unmatched_idx, ref_idx])
   

    note_alignments = np.array(note_alignments)
    note_alignments = note_alignments[np.argsort(note_alignments[:,0]),:]
    return note_alignments

def get_merging_idx(array, threshold = 2):
    array = array[np.argsort(array)]

    mergers = defaultdict(list)
    prev_e = array[0] - threshold * 2
    prev_i = -1
    for i, e in enumerate(array):
        if abs(e - prev_e) < threshold:
            mergers[prev_i].append(i)
        else:
            prev_e = e
            prev_i = i
    return mergers

def get_pitch_similarity_matrix(note_array, # pitch, onset
                                note_array_ref): # pitch, onset
    matrix = np.zeros((len(note_array), len(note_array_ref)))
    for id, note in enumerate(note_array):
        matrix[id, :] = np.exp(- 1 * np.abs(note_array_ref[:,0] - note[0]))
    return matrix

def get_onset_similarity_matrix(note_array, # pitch, onset
                                note_array_ref): # pitch, onset
    matrix = np.zeros((len(note_array), len(note_array_ref)))
    for id, note in enumerate(note_array):
        matrix[id, :] = np.exp(- 1 * np.abs(note_array_ref[:,1] - note[1]))
    return matrix

def get_input_to_ref_map(  note_array, # pitch, onset
                            note_array_ref,  # pitch, onset
                            alignment_idx,
                            merge_close_onsets = 5,
                            return_callable = True): # na_idx, na_ref_idx

        na_onset = note_array[alignment_idx[:,0],1]
        na_ref_onset = note_array_ref[alignment_idx[:,1],1]
        onsets = np.column_stack((na_onset, na_ref_onset))

        if merge_close_onsets:
            # onsets s1
            keep_idx = np.full(len(onsets), True)
            onsets = onsets[np.argsort(onsets[:,0]),:]
            merging_ids = get_merging_idx(onsets[:,0], threshold = merge_close_onsets)
            new_vals_list = list()
            for merging_idx in merging_ids.keys():
                mask = np.array([merging_idx] + merging_ids[merging_idx])
                new_vals = np.median(onsets[mask, :], axis = 0)
                new_vals_list.append(new_vals)
                keep_idx[mask] = False

            if len(new_vals_list) > 0:
                new_vals_array = np.row_stack(new_vals_list)
                onsets = np.concatenate((onsets[keep_idx], new_vals_array),axis = 0)

            # onsets s2
            keep_idx = np.full(len(onsets), True)
            onsets = onsets[np.argsort(onsets[:,1]),:]
            merging_ids = get_merging_idx(onsets[:,1], threshold = merge_close_onsets)
            new_vals_list = list()
            for merging_idx in merging_ids.keys():
                mask = np.array([merging_idx] + merging_ids[merging_idx])
                new_vals = np.median(onsets[mask, :], axis = 0)
                new_vals_list.append(new_vals)
                keep_idx[mask] = False

            if len(new_vals_list) > 0:
                new_vals_array = np.row_stack(new_vals_list)
                onsets = np.concatenate((onsets[keep_idx], new_vals_array),axis = 0)
            # sort again
            onsets = onsets[np.argsort(onsets[:,0]),:]
        
        if return_callable:
            input_to_ref_map = interp1d(onsets[:,0],
                                        onsets[:,1],
                                        kind = "linear",
                                        fill_value = "extrapolate")

            return input_to_ref_map
        else:
            return onsets

# path retrievers and processors

def get_path_from_confidence_matrix(mat, 
                                    directional_weights = np.array([1, 2, 1])):
    wdtw = WDTW(directional_weights = directional_weights)
    dmat = invert_matrix(mat)
    path = wdtw.from_distance_matrix(dmat)[0]
    return path

def get_local_path_from_confidence_matrix(confidence_matrix):
    path = get_path_from_confidence_matrix(confidence_matrix)
    (new_path, 
     starting_path, ending_path, 
     startpoints, endpoints) = get_path_endpoints(path, confidence_matrix.shape, cutoff=1)
    
    return (new_path, 
            starting_path, ending_path, 
            startpoints, endpoints)

def get_flex_path_from_confidence_matrix(confidence_matrix, 
                                         directional_weights = np.array([1, 2, 1])):

    buffer_from_conf_mat = min((confidence_matrix.shape[0],confidence_matrix.shape[1]))
    fdtw = FDTW(directional_weights = directional_weights, buffer = buffer_from_conf_mat - 20)
    dmat = invert_matrix(confidence_matrix)
    path = fdtw.from_distance_matrix(dmat)[0]
    (new_path, 
     starting_path, ending_path, 
     startpoints, endpoints) = get_path_endpoints_flexdtw(path, confidence_matrix.shape)
    
    return (new_path, 
            starting_path, ending_path, 
            startpoints, endpoints)

def get_path_endpoints_flexdtw(path, 
                               size):
    new_path = np.copy(path)

    startpoints = np.array(path[0,:])
    endpoints = np.array(path[-1,:]) - 1

    starting_path = np.array([])
    ending_path = np.array([])
    s1_exclusion_start = True
    s1_exclusion_end = True

    if startpoints[0] > 0:
        s1_exclusion_start = True
        starting_path = np.column_stack((np.arange(startpoints[0]), 
                                         np.zeros(startpoints[0])))

    elif startpoints[1] > 0:
        s1_exclusion_start = False
        starting_path = np.column_stack((np.zeros(startpoints[1]),
                                         np.arange(startpoints[1])))
    
    if endpoints[0] < size[0] - 1:
        s1_exclusion_end = True
        ending_path = np.column_stack((np.arange(endpoints[0] + 1, size[0]), 
                                         np.zeros(size[0] - endpoints[0] - 1)))

    elif endpoints[1] < size[1] - 1:
        s1_exclusion_end = True
        ending_path = np.column_stack((np.zeros(size[1] - endpoints[1] - 1),
                                       np.arange(endpoints[1] + 1, size[1])))

    return new_path, starting_path, ending_path, s1_exclusion_start, s1_exclusion_end

def get_path_endpoints(path, size, cutoff=1):
    new_path = np.copy(path)
    starting_path = np.array([])
    ending_path = np.array([])
    startpoints = np.array([0,0])
    endpoints = np.array(size) - 1

    left_path = path[:,1] < 1
    right_path = path[:,1] > size[1] - 2
    top_path = path[:,0] < 1
    bottom_path = path[:,0] > size[0] - 2

    new_path_mask_start = np.ones(len(path)) > 0
    new_path_mask_end = np.ones(len(path)) > 0

    s1_exclusion_start = True
    s1_exclusion_end = True

    if left_path.sum() > cutoff:
        # remove last path element from starting path
        last_start_entry_id = np.where(left_path == True)[0][-1]
        left_path[last_start_entry_id] = False
        # update output vars
        starting_path = path[left_path, :]
        new_path_mask_start = np.invert(left_path) 
        s1_exclusion_start = True
    elif top_path.sum() > cutoff:
        # remove last path element from starting path
        last_start_entry_id = np.where(top_path == True)[0][-1]
        top_path[last_start_entry_id] = False
        # update output vars
        starting_path = path[top_path, :]
        new_path_mask_start = np.invert(top_path) 
        s1_exclusion_start = False


    if right_path.sum() > cutoff:
        first_end_entry_id = np.where(right_path == True)[0][0]
        right_path[first_end_entry_id] = False
        # update output vars
        ending_path = path[right_path, :]
        new_path_mask_end = np.invert(right_path) 
        s1_exclusion_end = True

    elif bottom_path.sum() > cutoff:
        first_end_entry_id = np.where(bottom_path == True)[0][0]
        bottom_path[first_end_entry_id] = False
        # update output vars
        ending_path = path[bottom_path, :]
        new_path_mask_end = np.invert(bottom_path) 
        s1_exclusion_end = False

    new_path_mask = np.all((new_path_mask_start, new_path_mask_end), axis = 0)
    new_path = new_path[new_path_mask]
    return new_path, starting_path, ending_path, s1_exclusion_start, s1_exclusion_end

# full functions

def get_alignment_from_confidence_matrix(note_array,# pitch, onset
                                         note_array_ref,
                                         confidence_matrix):
    path = get_path_from_confidence_matrix(confidence_matrix)
    input_to_ref_map = get_input_to_ref_map(note_array,
                                            note_array_ref,
                                            path)
    alignment = get_note_matches(note_array,
                                note_array_ref,
                                input_to_ref_map,
                                onset_threshold = 500000,
                                unmatched_idx = 100000000)
    return alignment, path

def get_alignment_from_path(note_array,# pitch, onset
                            note_array_ref,
                            path):
    input_to_ref_map = get_input_to_ref_map(note_array,
                                            note_array_ref,
                                            path)
    alignment = get_note_matches(note_array,
                                note_array_ref,
                                input_to_ref_map,
                                onset_threshold = 500000,
                                unmatched_idx = 100000000)
    return alignment

if __name__ == "__main__":
    pass
