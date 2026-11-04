October 21<sup>st</sup> – Search Algorithms – Task 1 : Research how random search and evolutionary algorithms could be used to explore possible stabilizer codes 

Random search: samples random vectors in symplectic space(even-dimensional, all spaces in the same dimension are isomorphic, pauli operators mapped to binary vectors), must check if the chosen generators are commutable (AB = BA) 

Two pauli operators commute if their symplectic inner product is 0 

Evolutionary Algorithm: 

- Randomly generated stabilizer codes 

- Each is tested based on distance, code rate or the ability to minimize the undetectable error rate against a noise model 

- Distance: minimum number of physical qubit errors required to bypass the stabilizers 

   - 1) A code with distance d can correct any combination of up to [(d-1)/2] errors 

- Code Rate (k/n): ratio of logical qubits (k) to total physical qubits (n) 

   - 1) Goal is to maximize k/n 

   - 2) If random search evaluates 2 codes that have distance of d = 3, code A uses 7 physical qubits to store 1 logical and Code B uses 5 physical qubits to store 1. Code B is ranked higher and Is more efficient because we are wasting less physical qubit. 

- Fitness Function: 

   - 1) Search algorithm scores codes use fitness function 

Finally, cross over the best matrices 

How to Cross Over: 

- 1) Block Swapping: excellent X checking block from A and good Z checking block and form them into one 

- 2) Symplectic Subspace Splitting: algorithm takes a subset of commuting generators from A and B and does some linear algebra stuff 

Random Sequential Generation with Tracking Basis gemini generated: 

import numpy as np 

def symplectic_inner_product(v1, v2): 

""" 

Computes the symplectic inner product of two vectors in binary representation. v1, v2 are arrays of length 2n, where the first n elements are X, next n are Z. 

""" 

n = len(v1) // 2 x1, z1 = v1[:n], v1[n:] x2, z2 = v2[:n], v2[n:] # (X1 * Z2 + Z1 * X2) % 2 

return (np.dot(x1, z2) + np.dot(z1, x2)) % 2 

def generate_random_stabilizer_code(n, k): 

""" 

Hardcodes constraints to generate a valid, random stabilizer code with n physical qubits and k logical qubits. """ 

num_generators = n - k stabilizers = [] 

# We loop until we successfully collect (n - k) linearly independent, 

# mutually commuting generators. 

attempts = 0 

while len(stabilizers)  0: 

# Check linear independence in the binary vector space 

matrix_to_test = np.vstack([stabilizers, candidate]) 

# Calculate binary rank via GF(2) elimination or basic check for small n 

# (For exactness in large matrices, use row reduction over GF2) 

if np.linalg.matrix_rank(matrix_to_test.astype(float)) <= len(stabilizers): 

# continue 

stabilizers.append(candidate) 

if len(stabilizers) < num_generators: 

raise RuntimeError("Search timed out. Try fewer constraints or a smaller code.") 

return np.array(stabilizers) 

# Example Usage: Generate a random [[5, 1]] stabilizer code # 5 physical qubits, 1 logical qubit -> Requires 4 generators 

n, k = 5, 1 

H = generate_random_stabilizer_code(n, k) 

print(f"Generated a valid, hard-constrained H matrix of shape {H.shape}:\n") 

print("Left half (X block) | Right half (Z block)") 

print("-----------------------------------------") 

for row in H: 

print(f"{row[:n]}       | {row[n:]}") 

- Noise model: what kinds of errors happen and how frequently 

- Common types of noise models: depolarizing noise – the basic one 

- Depolarizing noise model: assume XYZ errors are all equally likely to happen to any qubit at any time 

- Biased noise model assumes one type of error is much more in common than others by a factor 

- Applies random pauli operators and then crosses the best performing ones 

Evolutionary Example gemini generated: 

pip install numpy 

import numpy as np 

# --- 1. CORE SYMPLECTIC UTILITIES --- 

def symplectic_inner_product(v1, v2): """Returns 0 if two Pauli vectors commute, 1 if they do not.""" n = len(v1) // 2 return (np.dot(v1[:n], v2[n:]) + np.dot(v1[n:], v2[:n])) % 2 

def generate_random_code(n, k): """Generates a single valid, mutually commuting stabilizer matrix.""" num_gens = n - k stabilizers = [] attempts = 0 while len(stabilizers) < num_gens and attempts < 2000: attempts += 1 candidate = np.random.randint(0, 2, size=2*n) if not np.any(candidate): continue if any(symplectic_inner_product(candidate, s) != 0 for s in stabilizers): continue # Check linear independence over GF(2) if len(stabilizers) > 0: test_mat = np.vstack([stabilizers, candidate]) if np.linalg.matrix_rank(test_mat.astype(float)) <= len(stabilizers): continue 

stabilizers.append(candidate) 

return np.array(stabilizers) 

# --- 2. THE FITNESS FUNCTION (NOISE MODEL EVALUATOR) --- 

def evaluate_fitness(H, n, num_simulations=200): 

""" 

Evaluates how well a code H protects against a Biased Z-Noise Model. Returns a score based on how often errors are successfully detected. """ 

# Noise parameters: Phase-flips (Z) are 10x more likely than bit-flips (X) p_x = 0.02 p_z = 0.20 

successful_detections = 0 

for _ in range(num_simulations): 

# Generate physical errors on 'n' qubits based on noise model probabilities error_x = (np.random.rand(n) < p_x).astype(int) error_z = (np.random.rand(n) < p_z).astype(int) error_vector = np.concatenate([error_x, error_z]) 

if not np.any(error_vector): 

successful_detections += 1  # No error occurred; perfect. 

continue 

# Calculate Error Syndrome: Multiply H by the error vector symplectically 

# Syndrome bit i = 0 if error commutes with stabilizer i, 1 if it flips it. 

syndrome = np.array([symplectic_inner_product(row, error_vector) for row in H]) 

# If syndrome is non-zero, the code successfully spotted that an error happened! 

if np.any(syndrome): 

successful_detections += 1 

# Fitness is the percentage of errors successfully caught by the stabilizers 

return successful_detections / num_simulations 

# --- 3. CROSSOVER OPERATOR (BREEDING PAPENTS) --- 

def crossover(parent_A, parent_B, n, k): 

""" 

Takes two valid parent codes and attempts to cross their structures. 

Uses an isotropic subspace stitching method to preserve commutation constraints. 

""" 

num_gens = n - k 

# Gather a pool of strong generators from both parents 

pool = np.vstack([parent_A, parent_B]) 

np.random.shuffle(pool) # Randomize row extraction order 

child_generators = [] 

for candidate in pool: 

if len(child_generators) == num_gens: 

break 

# Enforce strict commutation check for the child 

if any(symplectic_inner_product(candidate, s) != 0 for s in child_generators): 

continue 

# Enforce linear independence check 

if len(child_generators) > 0: 

test_mat = np.vstack([child_generators, candidate]) 

if np.linalg.matrix_rank(test_mat.astype(float)) <= len(child_generators): 

continue 

child_generators.append(candidate) 

# If the combined parents didn't yield a complete set, fill the gap with a random generator while len(child_generators) < num_gens: 

rand_gen = generate_random_code(n, n-1)[0] # get 1 row 

if all(symplectic_inner_product(rand_gen, s) == 0 for s in child_generators): 

child_generators.append(rand_gen) 

return np.array(child_generators) 

# --- 4. MAIN GENETIC LOOP --- 

def run_evolutionary_algorithm(n, k, pop_size=20, generations=10): 

print(f"Initializing population of {pop_size} random [[{n}, {k}]] stabilizer codes...") 

# Initialize a random starting population of valid matrices 

population = [generate_random_code(n, k) for _ in range(pop_size)] 

for gen in range(generations): 

# 1. Evaluate fitness for everyone 

scores = [evaluate_fitness(H, n) for H in population] 

# Sort population by highest fitness score 

sorted_indices = np.argsort(scores)[::-1] population = [population[i] for i in sorted_indices] scores = [scores[i] for i in sorted_indices] 

print(f"Generation {gen+1} | Best Error Detection Rate: {scores[0]:.2%}") 

# 2. Selection: Keep the top 25% of the population as elites num_elites = max(2, pop_size // 4) next_generation = population[:num_elites] 

# 3. Breeding / Crossover loop to fill the rest of the population 

while len(next_generation) < pop_size: # Pick two parents randomly from the top performing elites p1, p2 = np.random.choice(num_elites, size=2, replace=False) parent_A = population[p1] parent_B = population[p2] 

# Cross them together 

child = crossover(parent_A, parent_B, n, k) 

# Optional Mutation: 15% chance to inject a small structural mutation 

if np.random.rand() < 0.15: mutation_row = np.random.randint(0, len(child)) 

child[mutation_row] = (child[mutation_row] + np.random.randint(0, 2, size=2*n)) % 

2 

next_generation.append(child) 

population = next_generation 

return population[0], scores[0] 

# --- RUN EXECUTION --- 

# Optimize a [[7, 1]] code space for a heavily biased phase noise system 

best_code, best_score = run_evolutionary_algorithm(n=7, k=1, pop_size=30, generations=5) 

print(f"\nEvolution complete! Best discovered stabilizer matrix caught {best_score:.2%} of simulated hardware errors.") 

Task 2: Implement a basic random generator for sequences of Clifford gates (H, S, CNOT) Task 3: Look into simple mutation strategies, such as replacing, inserting, or removing a gate 

Task 4: Think about how we might compare and rank candidate codes once the evaluation system is available 

