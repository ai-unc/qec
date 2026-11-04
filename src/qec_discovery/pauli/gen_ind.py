from .commutation_check import commutation_check
import numpy as np

# input: (generated) set of stabilizer generators
# returns: False if any pair anticommutes or if any generator can be expressed as a combination of the others
def gen_ind(generators):
    m = len(generators)

    n = int(len(generators[0]) / 2)


    # commutation check
    for index in range(m - 1):
        # check all pairs by moving forward from index
        for j in range(index + 1, m - 1, 1):
            if(commutation_check(generators[index], generators[j]) == 1):
                return False

    # rref
    mat = np.array(generators)

    # set of pivots already recorded
    pivots = set()
    # remaining rows to reduce
    remaining = set(range(m))
    for index in range(2 * n):
        # pivot searching
        for i in range(m):
            if(mat[i][index] == 1):
                # pivot candidate found
                if(i in pivots):
                    # candidate is a previously used pivot, skip
                    continue

                # save pivot in set
                pivots.add(i)
                # we want to record the last row remaining to check for a 0 row
                if(len(remaining) != 1): remaining.remove(i)

                # ensure that every other row has 0 in the index position
                for j in range(m):
                    if(i == j):
                        continue
                    if(mat[j][index] == 1):
                        # GF(2)
                        mat[j] = (mat[j] + mat[i]) % 2
            else:
                # entire column is 0
                pass




    # zero row check, only need to check last entry in remaining row
    last = next(iter(remaining))
    return bool(mat[last].any())