import numpy as np

# vocab
# 0 PAD->padding, fills sequence to be some fixed len
# 1 BOS-> beginning of sequence
# 2 EOS-> end of sequence
# 3 SEP -> gap between the pauli generators
# The Pauli letters are as follows: 4 (I), 5 (X), 6 (Y), 7 (Z)
# Example: ["ZZI", "IZZ"] -> [1, 7, 7, 4, 3, 4, 7, 7, 2]
PAD = 0
BOS = 1
EOS = 2
SEP = 3
PAULI_AS_ID = {"I": 4, "X": 5, "Y": 6, "Z": 7}
ID_AS_PAULI = {4: "I", 5: "X", 6: "Y", 7: "Z"}
VOCAB_SIZE = 8


# INPUT: generators (list of Pauli strings of the same length
# RETRUNS: list of int
def encode(generators):
    if len(generators)== 0:
        raise ValueError("error: need at least one generator")
    n = len(generators[0])
    tokens = [BOS]
    for index, generator in enumerate(generators):
        if len(generator) != n:
            raise ValueError("generators must have same length")
        if index >0:
            tokens.append(SEP)
        for letter in generator:
            if letter not in PAULI_AS_ID:
                raise ValueError(str(letter) + " is not recognized")
            tokens.append(PAULI_AS_ID[letter])
    tokens.append(EOS)
    return tokens

# INPUT: tokens
# RETURNS: list of Pauli strings
def decode(tokens):
    generators = []
    current = ""
    for t in tokens:
        t = int(t)
        if t == BOS or t == PAD:
            continue
        if t == EOS:
            break
        if t == SEP:
            generators.append(current)
            current = ""
            continue
        if t not in ID_AS_PAULI:
            raise ValueError(str(t) + " isn't recognized")
        current = current + ID_AS_PAULI[t]
    if current != "":  # add last since no SEP
        generators.append(current)
    return generators
# INPUT: tokens, max_len -> max_len must be >= len tokens
# RETURNS: numpy int array of length max_len, PAD added at the end
def pad_tokens(tokens, max_len):
    if len(tokens)>max_len:
        raise ValueError("Input is longer than max length")
    padded = np.full(max_len, PAD)
    padded[:len(tokens)] = tokens
    return padded
