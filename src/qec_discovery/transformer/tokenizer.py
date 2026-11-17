from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qec_discovery.search.models import CircuitCandidate, GateOp


@dataclass(frozen=True)
class EncodedCircuit:
    ops: np.ndarray
    qa: np.ndarray
    qb: np.ndarray


class CliffordTokenizer:
    # Special tokens
    PAD = 0
    BOS = 1
    EOS = 2

    # Gate tokens
    OP_H = 3  # Hadamard gate
    OP_S = 4  # Phase gate
    OP_CNOT = 5  # CNOT gate
    OP_VOCAB_SIZE = 6

    # Qubit ids: 0=PAD, 1=BOS, 2=EOS, 3=UNUSED, then qubits start at 4
    Q_PAD = 0
    Q_BOS = 1
    Q_EOS = 2
    Q_UNUSED = 3
    Q_OFFSET = 4

    OP_TO_ID = {"H": OP_H, "S": OP_S, "CNOT": OP_CNOT}
    ID_TO_OP = {OP_H: "H", OP_S: "S", OP_CNOT: "CNOT"}

    def __init__(self, n_qubits=4, k_logical=2):
        self.n_qubits = n_qubits
        self.k_logical = k_logical
        self.max_gates = 8 * n_qubits
        self.max_seq_length = self.max_gates + 2  # BOS + gates + EOS
        self.qubit_vocab_size = self.Q_OFFSET + n_qubits

    def qubit_id(self, qubit: int):
        return self.Q_OFFSET + qubit

    def qubit_from_id(self, token):
        if token == self.Q_UNUSED:
            return None
        if self.Q_OFFSET <= token < self.qubit_vocab_size:
            return token - self.Q_OFFSET
        return None

    def encode_gate(self, gate: GateOp):
        op_id = self.OP_TO_ID[gate.op]
        if gate.op == "CNOT":
            return op_id, self.qubit_id(gate.qubits[0]), self.qubit_id(gate.qubits[1])
        return op_id, self.qubit_id(gate.qubits[0]), self.Q_UNUSED

    def decode_gate(self, op_id, qa_id, qb_id):
        if op_id in (self.PAD, self.BOS, self.EOS):
            return None
        op = self.ID_TO_OP.get(op_id)
        if op is None:
            return None
        qa = self.qubit_from_id(qa_id)
        if qa is None:
            return None
        if op in ("H", "S"):
            # Single-qubit gates must use UNUSED on the second operand.
            if qb_id != self.Q_UNUSED:
                return None
            return GateOp(op, (qa,))
        qb = self.qubit_from_id(qb_id)
        if qb is None or qa == qb:
            return None
        return GateOp("CNOT", (qa, qb))

    def encode(self, candidate):
        if candidate.n_qubits != self.n_qubits or candidate.k_logical != self.k_logical:
            raise ValueError("Candidate register does not match tokenizer")
        if len(candidate.gates) > self.max_gates:
            raise ValueError("Candidate exceeds max_gates")

        ops = [self.BOS]
        qa = [self.Q_BOS]
        qb = [self.Q_BOS]
        for gate in candidate.gates:
            o, a, b = self.encode_gate(gate)
            ops.append(o)
            qa.append(a)
            qb.append(b)
        ops.append(self.EOS)
        qa.append(self.Q_EOS)
        qb.append(self.Q_EOS)
        return EncodedCircuit(
            ops=np.asarray(ops, dtype=np.int32),
            qa=np.asarray(qa, dtype=np.int32),
            qb=np.asarray(qb, dtype=np.int32),
        )

    def decode(self, ops, qa, qb, n_qubits=None, k_logical=None):
        op_list = [int(x) for x in ops]
        qa_list = [int(x) for x in qa]
        qb_list = [int(x) for x in qb]
        if not (len(op_list) == len(qa_list) == len(qb_list)):
            raise ValueError("ops/qa/qb length mismatch")

        gates: list[GateOp] = []
        for op_id, qa_id, qb_id in zip(op_list, qa_list, qb_list):
            if op_id == self.BOS:
                continue
            if op_id == self.EOS:
                break
            if op_id == self.PAD:
                continue
            gate = self.decode_gate(op_id, qa_id, qb_id)
            if gate is not None:
                gates.append(gate)

        return CircuitCandidate(
            n_qubits if n_qubits is not None else self.n_qubits,
            k_logical if k_logical is not None else self.k_logical,
            tuple(gates),
        )

    def validity_masks(self, op_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Legal tokens for the next step given the chosen op."""
        op_mask = np.zeros(self.OP_VOCAB_SIZE, dtype=bool)
        op_mask[self.OP_H] = True
        op_mask[self.OP_S] = True
        if self.n_qubits >= 2:
            op_mask[self.OP_CNOT] = True
        op_mask[self.EOS] = True

        qa_mask = np.zeros(self.qubit_vocab_size, dtype=bool)
        qb_mask = np.zeros(self.qubit_vocab_size, dtype=bool)

        if op_id == self.EOS:
            qa_mask[self.Q_EOS] = True
            qb_mask[self.Q_EOS] = True
            return op_mask, qa_mask, qb_mask

        for q in range(self.n_qubits):
            qa_mask[self.qubit_id(q)] = True

        if op_id in (self.OP_H, self.OP_S):
            qb_mask[self.Q_UNUSED] = True
        elif op_id == self.OP_CNOT:
            for q in range(self.n_qubits):
                qb_mask[self.qubit_id(q)] = True
        # Unknown / special: allow nothing on qubits (caller should not sample).

        return op_mask, qa_mask, qb_mask

    def qb_mask_for_qa(self, op_id: int, qa_id: int) -> np.ndarray:
        _, _, qb_mask = self.validity_masks(op_id)  # freshly allocated, safe to mutate
        if op_id == self.OP_CNOT and self.Q_OFFSET <= qa_id < self.qubit_vocab_size:
            qb_mask[qa_id] = False
        return qb_mask