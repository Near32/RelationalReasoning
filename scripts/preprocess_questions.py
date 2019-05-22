import torch
import torch.nn as nn 
import torch.nn.functional as F 

import os
import json
import argparse 
import numpy as np 
import h5py

'''
    TOKENIZATION:
'''

SPECIAL_TOKENS={
    '<NULL>'    :0,
    '<START>'   :1,
    '<END>'     :2,
    '<UNK>'     :3    
}

def tokenize( string, delim=' ', punct_to_remove=None, punct_to_keep=None, add_start=True, add_end=True):
    if punct_to_keep is not None:
        for p in punct_to_keep:
            string = string.replace( p, '{}{}'.format(delim,p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            string = string.replace( p, '')

    tokens = string.split(delim)
    if add_start: tokens.insert(0, '<START>')
    if add_end: tokens.insert(-1, '<END>')

    return tokens 


'''
    VOCABULARY and MAPPING:
'''

def build_vocab(strings, min_token_count=1, delim=' ', punct_to_remove=None, punct_to_keep=None):
    tokens_count = {}
    '''
    Process each string's tokens:
    '''
    for string in strings:
        s_tokens = tokenize(string, delim, punct_to_keep, punct_to_remove, add_start=False, add_end=False)
        for token in s_tokens:
            if not(token in tokens_count):
                tokens_count[token] = 0
            tokens_count[token] += 1

    '''
    Map each token to an index, 
    while taking care of the SPECIAL_TOKENS 
    that are not included in the tokens_count:
    '''
    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx

    for token, count in sorted(tokens_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx 

def vocab_to_idx_encode(string, token_to_idx, allow_unk=True):
    seq_idx = []
    for token in string:
        if token in token_to_idx:
            seq_idx.append(token_to_idx[token])
        elif allow_unk:
            seq_idx.append(SPECIAL_TOKENS['<UNK>'])
        else:
            raise KeyError('Token {} is not in the vocabulary.'.format(token))

    return seq_idx

def invert_dict(d):
    return { v:k for k,v in d.items()}

def idx_to_vocab_decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens

    return delim.join(tokens) 

'''
    MAIN PROCESSING:
'''

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_json', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--output_vocab_json', default='')
parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--unk_threshold', type=int, default=1)
parser.add_argument('--encode_unk', type=int, default=1)


def main(args):
    if args.input_vocab_json == '' and args.output_vocab_json == '':
        print('Need to provide either --input_vocab_json or --output_vocab_json.')
        return 

    with open(args.input_data_json, 'r') as f:
        data = json.load(f)
        answers_questions = data['questions']


    '''
        BUILD VOCABS:
    '''
    if args.input_vocab_json == '':
        print('Building vocabulary...')
        if 'answer' in answers_questions[0]:
            answer_token_to_idx = build_vocab( [q['answer'] for q in answers_questions],
                                                min_token_count=1,
                                                delim=' ',
                                                punct_to_remove=None,
                                                punct_to_keep=None
                                                )
        question_token_to_idx = build_vocab( [q['question'] for q in answers_questions],
                                            min_token_count=args.unk_threshold,
                                            delim=' ',
                                            punct_to_remove=['?','.'],
                                            punct_to_keep=[';',',']
                                            )

        vocabs = {'question_vocab': question_token_to_idx,
                    'answer_vocab': answer_token_to_idx
                    }
    else:
        print('Loading vocabulary...')
        with open(args.input_vocab_json, 'r') as f:
            vocabs = json.load(f)


    print('Dumping vocabulary...')
    if args.output_vocab_json != '':
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocabs, f)

    '''
        ENCODE VOCABS:
    '''
    print('Encoding vocabulary...')
    encoded_questions = []
    encoded_answers = []
    question_family_idxs = []
    orig_idxs = []
    image_idxs = []
    for orig_idx, q in enumerate(answers_questions):
        orig_idxs.append(orig_idx)
        image_idxs.append(q['image_index'])
        if 'question_family_index' in q: question_family_idxs.append(q['question_family_index'])
        
        question = q['question']
        question_tokens = tokenize( question, 
                                    delim=' ', 
                                    punct_to_remove=['?','.'], 
                                    punct_to_keep=[';',','], 
                                    add_start=True, 
                                    add_end=True
                                    )
        encoded_question = vocab_to_idx_encode(question_tokens, 
                                            vocabs['question_vocab'],
                                            allow_unk= (args.encode_unk == 1)
                                            )
        encoded_questions.append(encoded_question)

        if 'answer' in q:
            encoded_answers.append( vocabs['answer_vocab'][q['answer']] )


    '''
        PADDING:
    '''
    print('Padding...')
    max_len_question = max(len(eq) for eq in encoded_questions)
    for eq in encoded_questions:
        while len(eq) < max_len_question:
            eq.append( vocabs['question_vocab']['<NULL>'])


    '''
        DUMPING:
    '''
    print('Dumping encoded vocabularies...')
    encoded_questions = np.asarray(encoded_questions, dtype=np.int32)
    encoded_answers = np.asarray(encoded_answers, dtype=np.int32)
    question_family_idxs = np.asarray(question_family_idxs, dtype=np.int32)
    image_idxs = np.asarray(image_idxs, dtype=np.int32)
    orig_idxs = np.asarray(orig_idxs, dtype=np.int32)
    with h5py.File(args.output_h5_file, 'w') as f:
        f.create_dataset('questions', data=encoded_questions)
        f.create_dataset('answers', data=encoded_answers)
        f.create_dataset('question_families', data=question_family_idxs)
        f.create_dataset('image_idxs', data=image_idxs)
        f.create_dataset('orig_idxs', data=orig_idxs)
    print('DONE.')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)