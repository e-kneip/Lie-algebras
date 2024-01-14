"""Useful functions for constructing Lie algebras."""

import numpy as np
from numbers import Number

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

def a_comm(A: np.ndarray, B: np.ndarray):
    """
    Calculate anticommutator of operators: [A, B] = AB+BA.

    Parameters
    ----------
    A : np.ndarray
        First operator
    B : np.ndarray
        Second operator

    Returns
    -------
    C : np.ndarray
        Anticommutation of A and B, [A, B] = AB+BA
    """
    C = A @ B + B @ A
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

def decompose(M):
    """
    Decompose matrix M into a linear combination of tensors of Pauli matrices.
    
    Parameters
    ----------
    M : np.ndarray
        Matrix to decompose
        
    Returns
    -------
    coeff : list
        List of coefficients of Pauli basis
    b2 : list
        List of Pauli matrix tensors forming basis
    """
    # define basis matrices
    n = int(np.log2(M.shape[0]))
    # construct basis
    b1 = [I, X, Y, Z]
    b2 = [I, X, Y, Z]
    for i in range(n-1):
        b2 = np.kron(b1, b2)
    
    # find coefficients
    coeff = []
    for i in range(len(b2)):
        coeff.append(1/2**n * np.trace(b2[i].conj().T@M))

    return coeff, b2


# keeping everything in Pauli decomposition from here on...
class Pauli:
    """
    Class for tensor products of Pauli matrices.
    """

    def __init__(self, coeff: Number = 1, decomp: list = []):
        """
        Initialise Pauli tensor product.

        Parameters
        ----------
        decomp : list
            List of Pauli matrices in tensor product
        coeff : float or complex
            Coefficient of tensor product
        """
        # check correct input
        if not isinstance(coeff, Number):
            raise TypeError(f"Coefficient {coeff} must be a number.")
        if not isinstance(decomp, list):
            raise TypeError(f"Decomposition {decomp} must be a list.")
        for op in decomp:
            if not isinstance(op, np.ndarray):
                raise TypeError(f"Decomposition must be a list of numpy arrays, but {op} is not.")
            elif not (op == I).all() and not (op == X).all() and not (op == Y).all() and not (op == Z).all():
                raise ValueError(f"Decomposition must be a list of Pauli matrices, but {op} is not.")

        # create quick storage of decomposition
        q_decomp = np.array([])
        for op in decomp:
            if (op == I).all():
                q_decomp = np.append(q_decomp, 0)
            elif (op == X).all():
                q_decomp = np.append(q_decomp, 1)
            elif (op == Y).all():
                q_decomp = np.append(q_decomp, 2)
            else:
                q_decomp = np.append(q_decomp, 3)

        # set attributes
        self.coeff = coeff
        self.decomp = decomp
        self.q_decomp = q_decomp

    def ldim(self):
        """
        Get log2(dimension) of Pauli tensor product.

        Returns
        -------
        dim : int
            log base 2 of the dimension of Pauli tensor product
        """
        return len(self.decomp)
    
    def __eq__(self, other):
        """
        Check if Pauli tensor products are equal.

        Parameters
        ----------
        other : Pauli
            Pauli tensor product to compare with

        Returns
        -------
        bool
            True if equal, False otherwise
        """
        return isinstance(other, Pauli) and np.array_equal(self.decomp, other.decomp) and self.coeff == other.coeff
    
    def __str__(self):
        """
        Print Pauli tensor product.

        Returns
        -------
        str
            String representation of Pauli tensor product
        """
        q_decomp = self.q_decomp
        str_decomp = []
        for i in range(self.ldim()):
            if q_decomp[i] == 0:
                str_decomp.append("I")
            elif q_decomp[i] == 1:
                str_decomp.append("X")
            elif q_decomp[i] == 2:
                str_decomp.append("Y")
            else:
                str_decomp.append("Z")
        return f"{self.coeff} * {' x '.join(str_decomp)}"
    
    def __repr__(self):
        """
        Print Pauli tensor product representation.

        Returns
        -------
        str
            String representation of Pauli tensor product
        """
        q_decomp = self.q_decomp
        str_decomp = []
        for i in range(self.ldim()):
            if q_decomp[i] == 0:
                str_decomp.append("I")
            elif q_decomp[i] == 1:
                str_decomp.append("X")
            elif q_decomp[i] == 2:
                str_decomp.append("Y")
            else:
                str_decomp.append("Z")
        return "Pauli(" + str(self.coeff) + ", " + "[" + ", ".join(str_decomp) + "]" + ")"

# commutator/anti-commutator lookup table
look_up = np.array([[[2, 0], [2j, 1], [-2j, 2], [2, 1]], [[-2j, 3], [2, 0], [2j, 1], [2, 2]], [[2j, 2], [-2j, 1], [2, 0], [2, 3]], [[2, 1], [2, 2], [2, 3], [2, 0]]])

def p_ao_comm(A: Pauli, B: Pauli):
    """
    Calculate commutator and anticommutator of Pauli tensor products.
    
    Parameters
    ----------
    A : Pauli
        First Pauli tensor product
    B : Pauli
        Second Pauli tensor product
    
    Returns
    -------
    C : Pauli
        Sum of commutator and anticommutator of A and B
    aoc : bool
        True if anticommutator, False if commutator
    """
    a_decomp, b_decomp = A.decomp, B.decomp
    aoc = False

    # commutator of every pair of Pauli matrices in decompositions
    commutator = np.einsum("ijk, ikl -> ijl", a_decomp, b_decomp, optimize=True) - np.einsum("ijk, ikl -> ijl", b_decomp, a_decomp, optimize=True)
    commutator = np.array(commutator / 2j)

    # anti-commutator of every pair of Pauli matrices in decompositions
    anticommutator = np.einsum("ijk, ikl -> ijl", a_decomp, b_decomp, optimize=True) + np.einsum("ijk, ikl -> ijl", b_decomp, a_decomp, optimize=True)
    anticommutator = np.array(anticommutator / 2)

    # check how many commutators are non-zero and correct sign
    n = 0
    sgn = 1
    for i in range(len(commutator)):
        ind1 = commutator[i][0][0]
        ind2 = commutator[i][1][0]
        if not (ind1 == 0 and ind2 == 0):
            n += 1
            if (ind1 + ind2).real + (ind1 + ind2).imag < 0:
                commutator[i] = -commutator[i]
                sgn *= -1

    # odd n => imaginary coefficient => commutator or even n => real coefficient => anticommutator
    if n%2 == 0:
        aoc = True

    decomp = list(commutator + anticommutator)
    C = Pauli(decomp, 2 * sgn * A.coeff * B.coeff * 1j**n)

    return C, aoc

def p_comm(A: Pauli, B: Pauli):
    """
    Calculate commutator of Pauli tensor products.

    Parameters
    ----------
    A : Pauli
        First Pauli tensor product
    B : Pauli
        Second Pauli tensor product

    Returns
    -------
    C : Pauli
        Commutator of A and B
    """
    C, aoc = p_ao_comm(A, B)
    if aoc:
        return Pauli([np.zeros((2, 2)) for i in A.ldim()])
    else:
        return C

def p_a_comm(A: Pauli, B: Pauli):
    """
    Calculate anticommutator of Pauli tensor products.

    Parameters
    ----------
    A : Pauli
        First Pauli tensor product
    B : Pauli
        Second Pauli tensor product

    Returns
    -------
    C : Pauli
        Anticommutator of A and B
    """
    C, aoc = p_ao_comm(A, B)
    if aoc:
        return C
    else:
        return Pauli([np.zeros((2, 2)) for i in A.ldim()])
