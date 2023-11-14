"""Useful functions for constructing Lie algebras."""

import numpy as np

# some useful operators
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])


class MaxOperatorsError(Exception):
    pass


class LinearIndependenceError(Exception):
    pass


class LieAlgebraError(Exception):
    pass


def comm(A: np.ndarray, B: np.ndarray):
    """
    Calculate commutator of operators: [A, B] = AB-BA.

    Parameters
    ----------
    A : np.ndarray
        First operator
    B : np.ndarray
        Second operator

    Returns
    -------
    C : np.ndarray
        Commutation of A and B, [A, B] = AB-BA
    """
    C = A @ B - B @ A
    return C


def tensor(W: np.ndarray, dim: int, start: int, end: int = 0):
    """
    Return tensor product of dim operators acting with W on qubits [start, end] and trivially elsewhere.

    Parameters
    ----------
    W : np.ndarray
        Operator acting on quibits [start, end]
    dim : int
        Number of qubits in system
    start : int
        First non-trivial operator
    end : int
        Final non-trivial operator

    Returns
    -------
    T : np.ndarray
        Tensor product acting with W on qubits [start, end] and trivially elsewhere
    """
    if not end:
        end = start

    op = [I for i in range(dim)]

    for i in range(start, end + 1):
        op[i] = W

    T = op[0]
    for i in range(1, dim):
        T = np.kron(T, op[i])

    return T


def lin_ind(Ops: list):
    """
    Determines whether a list of operators are linearly independent.

    Parameters
    ----------
    Ops : list
        List of operators with equal dimension

    Returns
    -------
    ind : bool
        List of operators are linearly independent, true or false
    """
    # matrix such that each column corresponds to one operator
    L = np.array([np.squeeze(op.reshape((1, -1))) for op in Ops]).T

    rank = np.linalg.matrix_rank(L)
    ind = rank == len(Ops)
    return ind


def complete_algebra_inner(Ops: list, start: int):
    """Find all linearly independent operators from commutations of operators in Ops[0:len(Ops)] with operators in Ops[start:len(Ops)]."""
    new_Ops = Ops.copy()
    for i in range(len(Ops)):
        for j in range(max(i, start), len(Ops)):
            new_op = comm(new_Ops[i], new_Ops[j])
            new_Ops.append(new_op)
            if not lin_ind(new_Ops):
                new_Ops.pop()
    return new_Ops


def complete_algebra(Ops: list, max: int, start: int = 0):
    """
    Find closed Lie algebra given initial set of operators.

    Parameters
    ----------
    Ops : list
        List of initial operators in Lie algebra
    max : int
        Cut off after max number of operators in Lie algebra found
    start : int
        Operator index to start verifying closedness, ie every commutation with operators before start already accounted

    Returns
    -------
    new_Ops : list
        List of operators in completed Lie algebra
    """
    if not lin_ind(Ops):
        raise LinearIndependenceError(
            "Given operators are not linearly independent."
        )

    old_Ops = Ops.copy()

    while True:
        # find new set of linearly independent operators to extend old_Ops
        new_Ops = complete_algebra_inner(old_Ops, start)

        # number of new operators added
        added_ops = len(new_Ops) - len(old_Ops)

        # if no new operators found, algebra is complete
        if added_ops == 0:
            return new_Ops
        else:
            start = len(old_Ops)
            old_Ops = new_Ops

        # stop if maximum operators in algebra reached
        if len(old_Ops) > max:
            raise MaxOperatorsError(
                f"Maximum of {max} operators in uncomplete algebra reached."
            )


def find_algebra(Op_0: list, Op_1: list, max: int):
    """
    Extend operators Op_0 to include commutations with operators in Op_1.

    Parameters
    ----------
    Op_0 : list
        List of operators to extend (defines invariants)
    Op_1 : list
        List of operators to extend Op_0 with commutations of (defines H)
    max : int
        Cut off after max number of operators in Lie algebra found

    Returns
    -------
    Lie_alg : list
        List of operators in extended Lie algebra
    """
    Ops = Op_0.copy()
    # append every commutation that is linearly independent
    for i in range(len(Op_0)):
        for j in range(len(Op_1)):
            new_op = comm(Op_0[i], Op_1[j])
            Ops.append(new_op)
            if not lin_ind(Ops):
                Ops.pop()
    # complete the algebra
    Lie_alg = complete_algebra(Ops, max)
    return Lie_alg
