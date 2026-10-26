def model(error, p):
    """Based on Textbook and Meeting Notes, the distribution of errors as dictionaries """
    if error == "bit_flip":
        return {"I": 1-p, "X": p, "Y": 0, "Z": 0}
    if error == "phase_flip":
        return {"I": 1-p, "X": 0, "Y": 0, "Z": p}

    if error == "depolarizing":
        return {"I": 1-p, "X": p/3, "Y": p/3, "Z": p/3}

    raise ValueError("Unknown noise model")

def error_probability(error_string, noise_model, p):
    probability_distribution = model(noise_model, p)
    # print(probability_distribution)

    total_prob = 1
    for pauli in error_string:
        total_prob = total_prob * probability_distribution[pauli]

    return total_prob