
from pair1 import commutation_check

# input: (generated) set of stabilizer generators
# returns: False if any pair anticommutes or if any generator can be expressed as a combination of the others
def gen_ind(generators):
    n = len(generators)

    for i in range(n - 1):
        for j in range(i + 1, n, 1):
            if(commutation_check(generators[i], generators[j]) == 1):
                return False

    for generator in generators:
        # TODO: RREF in GF(2) for binary vectors to check equivalence
        pass

    return True


    

    
