"""
Generate NELL-Wordnet entity dictionary
"""
def generate_vocab_dict(file_name):
    #construct vocab dictionary from Nell knowledge base.
    w2i_dict, i2w_dict = {}, {}
    w2i_dict[0] = 'UNK'
    i2w_dict['UNK'] = 0
    with open(file_name, 'r') as f:
            vocab = f.readlines()
            for idx, row in enumerate(vocab):
                #extract the actual concept and its generalization.
                #and correct spaces.
                row = row.strip()
                row = row.replace('_', ' ')
                row = row.split(':')
                word = row[-1]
                try:
                    w2i_dict[word] = idx+1
                    i2w_dict[idx+1] = word
                except KeyError:
                    print('fuck!')
    return w2i_dict, i2w_dict
