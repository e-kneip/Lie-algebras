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


def commutator(A: np.ndarray, B: np.ndarray):
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

def anticommutator(A: np.ndarray, B: np.ndarray):
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

def tensor_product(W: np.ndarray, dim: int, start: int, end: int = 0):
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


def linear_independence(Ops: list):
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
            new_op = commutator(new_Ops[i], new_Ops[j])
            new_Ops.append(new_op)
            if not linear_independence(new_Ops):
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
    if not linear_independence(Ops):
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
            new_op = commutator(Op_0[i], Op_1[j])
            Ops.append(new_op)
            if not linear_independence(Ops):
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
    """Class for tensor products of Pauli matrices."""

    def __init__(self, decomp: list = []):
        """
        Initialise Pauli tensor product.

        Parameters
        ----------
        decomp : list
            List of Pauli matrices in tensor product
        """
        # check correct input
        if not isinstance(decomp, list):
            raise TypeError(f"Decomposition must be a list, but {decomp} is not.")
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
        self.decomp = decomp
        self.q_decomp = q_decomp

    def __len__(self):
        """
        Get number of Paulis in tensor product, or log2(dimension) of Pauli tensor product.

        Returns
        -------
        dim : int
            Number of Paulis in tensor product
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
        return isinstance(other, Pauli) and np.array_equal(self.q_decomp, other.q_decomp)

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
        for i in range(len(self)):
            if q_decomp[i] == 0:
                str_decomp.append("I")
            elif q_decomp[i] == 1:
                str_decomp.append("X")
            elif q_decomp[i] == 2:
                str_decomp.append("Y")
            else:
                str_decomp.append("Z")
        return ' x '.join(str_decomp)

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
        for i in range(len(self)):
            if q_decomp[i] == 0:
                str_decomp.append("I")
            elif q_decomp[i] == 1:
                str_decomp.append("X")
            elif q_decomp[i] == 2:
                str_decomp.append("Y")
            else:
                str_decomp.append("Z")
        return "Pauli([" + ", ".join(str_decomp) + "])"

    def __add__(self, other):
        """
        Add Pauli tensor products.

        Parameters
        ----------
        other : Pauli
            Pauli tensor product to add

        Returns
        -------
        SuperPauli
            Sum of Pauli tensor products
        """
        if isinstance(other, Pauli):
            if self == other:
                return SuperPauli([(2, self)])
            else:
                return SuperPauli([(1, self), (1, other)])
        if isinstance(other, SuperPauli):
            return other.__add__(self)
        else:
            raise TypeError("Cannot add Pauli to " + type(other).__name__ + ".")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract Pauli tensor products.

        Parameters
        ----------
        other : Pauli
            Pauli tensor product to subtract

        Returns
        -------
        SuperPauli
            Difference of Pauli tensor products
        """
        if isinstance(other, Pauli):
            if self == other:
                return SuperPauli([])
            else:
                return SuperPauli([(1, self), (-1, other)])
        else:
            raise TypeError("Cannot subtract " + type(other).__name__ + " from Pauli.")

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        """
        Multiply Pauli tensor product by scalar.

        Parameters
        ----------
        other : Number
            Scalar to multiply by

        Returns
        -------
        SuperPauli :
            Multiplication of Pauli tensor product by scalar
        """
        if isinstance(other, Number):
            return SuperPauli([(other, self)])
        else:
            raise TypeError("Cannot multiply Pauli by " + type(other).__name__ + ".")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return SuperPauli([(-1, self)])

class SuperPauli:
    """Class for superposition of Pauli tensor products."""

    def __init__(self, paulis: list = []):
        """
        Initialise Pauli tensor product superposition.

        Parameters
        ----------
        paulis : list
            List of tuples (coefficient, Pauli) for each Pauli tensor product in superposition
        """
        coeff_list = []
        pauli_list = []
        new_paulis = []
        # check correct input
        if not isinstance(paulis, list):
            raise TypeError(f"Decomposition {paulis} must be a list.")
        for op in paulis:
            if not isinstance(op, tuple) or not len(op) == 2:
                raise TypeError(f"Decomposition must be a list of length 2 tuples, but {op} is not.")
            i, j = op
            if not isinstance(i, Number):
                raise TypeError(f"Coefficient {i} must be a number.")
            if not isinstance(j, Pauli):
                raise TypeError(f"Pauli {j} must be a Pauli tensor product.")
            if i != 0:
                new_paulis.append(op)
                coeff_list.append(i)
                pauli_list.append(j)

        # set attributes
        self.paulis = new_paulis
        self.coeff_list = coeff_list
        self.pauli_list = pauli_list

    def __len__(self):
        """
        Get number of Pauli tensor products in superposition.

        Returns
        -------
        int
            Number of Pauli tensor products in superposition
        """
        return len(self.paulis)

    def __str__(self):
        """
        Print Pauli tensor product.

        Returns
        -------
        str
            String representation of Pauli tensor product
        """
        paulis = self.paulis
        coeff_list, pauli_list = self.coeff_list, self.pauli_list
        return f" + ".join([f"{coeff_list[i]} * {pauli_list[i]}" for i in range(len(paulis))])
    
    def __repr__(self):
        """
        Print Pauli tensor product representation.

        Returns
        -------
        str
            String representation of Pauli tensor product
        """
        paulis = self.paulis
        coeff_list, pauli_list = self.coeff_list, self.pauli_list
        return "SuperPauli([" + ", ".join([f"({coeff_list[i]}, {pauli_list[i].__repr__()})" for i in range(len(paulis))]) + "])"

    def __add__(self, other):
        """
        Add Pauli tensor product superpositions.

        Parameters
        ----------
        other : SuperPauli
            Pauli tensor product superposition to add

        Returns
        -------
        SuperPauli
            Sum of Pauli tensor product superpositions
        """
        if isinstance(other, Pauli):
            other = SuperPauli([(1, other)])
        if isinstance(other, SuperPauli):
            self_pauli_list, self_coeff_list = self.pauli_list, self.coeff_list
            other_pauli_list, other_coeff_list = other.pauli_list, other.coeff_list
            new_pauli_list, new_coeff_list = self_pauli_list.copy(), self_coeff_list.copy()
            for i in range(len(other)):
                if other_pauli_list[i] in self_pauli_list:
                    new_coeff_list[self_pauli_list.index(other_pauli_list[i])] += other_coeff_list[i]
                else:
                    new_pauli_list.append(other_pauli_list[i])
                    new_coeff_list.append(other_coeff_list[i])
            return SuperPauli(list(zip(new_coeff_list, new_pauli_list)))
        else:
            raise TypeError("Cannot add SuperPauli to " + type(other).__name__ + ".")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract Pauli tensor product superpositions.

        Parameters
        ----------
        other : SuperPauli
            Pauli tensor product superposition to subtract

        Returns
        -------
        SuperPauli
            Difference of Pauli tensor product superpositions
        """
        if isinstance(other, Pauli):
            other = SuperPauli([(1, other)])
        if isinstance(other, SuperPauli):
            self_pauli_list, self_coeff_list = self.pauli_list, self.coeff_list
            other_pauli_list, other_coeff_list = other.pauli_list, other.coeff_list
            new_pauli_list, new_coeff_list = self_pauli_list.copy(), self_coeff_list.copy()
            for i in range(len(other)):
                if other_pauli_list[i] in self_pauli_list:
                    new_coeff_list[self_pauli_list.index(other_pauli_list[i])] -= other_coeff_list[i]
                else:
                    new_pauli_list.append(other_pauli_list[i])
                    new_coeff_list.append(-other_coeff_list[i])
            return SuperPauli(list(zip(new_coeff_list, new_pauli_list)))
        else:
            raise TypeError("Cannot subtract " + type(other).__name__ + " from SuperPauli.")

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        """
        Multiply Pauli tensor product superposition by scalar.

        Parameters
        ----------
        other : Number
            Scalar to multiply by

        Returns
        -------
        SuperPauli
            Multiplication of Pauli tensor product superposition by scalar
        """
        if isinstance(other, Number):
            return SuperPauli([(other * self.coeff_list[i], self.pauli_list[i]) for i in range(len(self))])
        else:
            raise TypeError("Cannot multiply SuperPauli by " + type(other).__name__ + ".")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return -1 * self

# commutator/anti-commutator lookup table
look_up = np.array([[[1, 0], [1, 1], [1, 2], [1, 3]], [[1, 1], [1, 0], [1j, 3], [-1j, 2]], [[1, 2], [-1j, 3], [1, 0], [1j, 1]], [[1, 3], [1j, 2], [-1j, 1], [1, 0]]])

def acomm_comm(A: Pauli, B: Pauli):
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
    C : SuperPauli
        Commutator or anticommutator of A and B, based on aoc
    aoc : bool
        True if anticommutator, False if commutator
    """
    a_q_decomp, b_q_decomp = A.q_decomp, B.q_decomp
    aoc = True
    c_q_decomp = np.array([0.0 + 0.0j for i in range(len(A))])
    coeff = 1

    # use lookup to commutate/anticommutate
    for i in range(4):
        for j in range(4):
            ind = np.where((a_q_decomp == i) & (b_q_decomp == j))[0]
            c_q_decomp[ind] = look_up[i, j, 1]
            coeff *= look_up[i, j, 0]**len(ind)

    decomp = []
    for op in c_q_decomp:
        if op == 0:
            decomp.append(I)
        elif op == 1:
            decomp.append(X)
        elif op == 2:
            decomp.append(Y)
        else:
            decomp.append(Z)

    # coeff imaginary => commutator
    if coeff.imag != 0:
        aoc = False

    C = SuperPauli([(2 * coeff, Pauli(decomp))])

    return C, aoc

def comm(A: Pauli or SuperPauli, B: Pauli or SuperPauli):
    """
    Calculate commutator of Pauli tensor products.

    Parameters
    ----------
    A : Pauli or SuperPauli
        First superposition of Pauli tensor products
    B : Pauli or SuperPauli
        Second superposition of Pauli tensor products

    Returns
    -------
    C : SuperPauli
        Commutator of A and B
    """
    if isinstance(A, Pauli):
        A = SuperPauli([(1, A)])
    if isinstance(B, Pauli):
        B = SuperPauli([(1, B)])

    a_pauli_list, b_pauli_list = A.pauli_list, B.pauli_list
    a_coeff_list, b_coeff_list = A.coeff_list, B.coeff_list
    basis = []
    coeffs = []

    # calculate commutator of each pair of Pauli tensor products
    for i in range(len(A)):
        for j in range(len(B)):
            C, aoc = acomm_comm(a_pauli_list[i], b_pauli_list[j])
            if not aoc:
                if C.pauli_list[0] in basis:
                    coeffs[basis.index(C.pauli_list[0])] += a_coeff_list[i] * b_coeff_list[j] * C.coeff_list[0]
                else:
                    basis.append(C.pauli_list[0])
                    coeffs.append(a_coeff_list[i] * b_coeff_list[j] * C.coeff_list[0])

    paulis = zip(coeffs, basis)

    return SuperPauli(list(paulis))

def acomm(A: Pauli or SuperPauli, B: Pauli or SuperPauli):
    """
    Calculate anticommutator of Pauli tensor products.

    Parameters
    ----------
    A : Pauli or SuperPauli
        First superposition of Pauli tensor products
    B : Pauli or SuperPauli
        Second superposition of Pauli tensor products

    Returns
    -------
    C : SuperPauli
        Anticommutator of A and B
    """
    if isinstance(A, Pauli):
        A = SuperPauli([(1, A)])
    if isinstance(B, Pauli):
        B = SuperPauli([(1, B)])

    a_pauli_list, b_pauli_list = A.pauli_list, B.pauli_list
    a_coeff_list, b_coeff_list = A.coeff_list, B.coeff_list
    basis = []
    coeffs = []

    # calculate commutator of each pair of Pauli tensor products
    for i in range(len(A)):
        for j in range(len(B)):
            C, aoc = acomm_comm(a_pauli_list[i], b_pauli_list[j])
            if aoc:
                if C.pauli_list[0] in basis:
                    coeffs[basis.index(C.pauli_list[0])] += a_coeff_list[i] * b_coeff_list[j] * C.coeff_list[0]
                else:
                    basis.append(C.pauli_list[0])
                    coeffs.append(a_coeff_list[i] * b_coeff_list[j] * C.coeff_list[0])

    paulis = zip(coeffs, basis)

    return SuperPauli(list(paulis))

def lin_ind(paulis: list, n: int = 0, old_basis: list = [], old_pauli_vecs: list = []):
    """
    Determines whether a list of SuperPauli tensor products are linearly independent.

    Parameters
    ----------
    paulis : list
        List of SuperPauli tensor products
    n : int
        Number of SuperPaulis already known to be linearly independent
    old_basis : list
        List of Paulis forming basis for first n SuperPaulis
    old_pauli_vecs : list
        List of first n SuperPaulis decomposed using old_basis

    Returns
    -------
    ind : bool
        List of Pauli tensor products are linearly independent, true or false
    basis : list
        List of Paulis forming basis for all SuperPaulis
    vectors : list
        List of SuperPaulis decomposed using basis
    """
    basis = old_basis.copy()
    for i, pauli in enumerate(paulis[n:]):
        if isinstance(pauli, Pauli):
            paulis[i+n] = SuperPauli([(1, pauli)])
            pauli = SuperPauli([(1, pauli)])
        for op in pauli.pauli_list:
            if op not in basis:
                basis.append(op)

    vectors = old_pauli_vecs.copy()
    to_add = [0 for i in range(len(basis) - len(old_basis))]
    for j, pauli in enumerate(paulis):
        if j<n:
            vectors[j] += to_add
        else:    
            vector = [0 for i in range(len(basis))]
            for op, coef in zip(pauli.pauli_list, pauli.coeff_list):
                vector[basis.index(op)] = coef
            vectors.append(vector)

    rank = np.linalg.matrix_rank(np.array(vectors))
    ind = rank == len(paulis)
    return ind, basis, vectors

def pauli_complete_algebra_inner(Ops: list, start: int, n: int = 0, old_basis: list = [], old_pauli_vecs: list = []):
    """
    Find all linearly independent operators from commutations of operators in Ops[0:len(Ops)] with operators in Ops[start:len(Ops)], using Pauli decompositions.
    """
    new_Ops = Ops.copy()
    for i in range(len(Ops)):
        for j in range(max(i+1, start), len(Ops)):
            new_op = comm(new_Ops[i], new_Ops[j])
            new_Ops.append(new_op)
            lind, old_basis, old_pauli_vecs = lin_ind(new_Ops, n, old_basis, old_pauli_vecs)
            if not lind:
                new_Ops.pop()
                old_pauli_vecs.pop()
            n = len(old_pauli_vecs)
    return new_Ops, n, old_basis, old_pauli_vecs


def pauli_complete_algebra(Ops: list, max: int, start: int = 0):
    """
    Find closed Lie algebra given initial set of operators, using Pauli decompositions.

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
    if not isinstance(Ops, list):
        raise TypeError(f"Ops must be a list, but {Ops} is not.")
    for i, op in enumerate(Ops):
        if not isinstance(op, Pauli) and not isinstance(op, SuperPauli):
            raise TypeError(f"Ops must be a list of Pauli tensor products (Pauli or SuperPauli), but {op} is not.")
        if isinstance(op, Pauli):
            Ops[i] = SuperPauli([(1, op)])
    if not lin_ind(Ops)[0]:
        raise LinearIndependenceError(
            "Given operators are not linearly independent."
        )

    old_Ops = Ops.copy()
    old_basis = []
    old_pauli_vecs = []
    n = 0

    while True:
        # find new set of linearly independent operators to extend old_Ops
        new_Ops, n, old_basis, old_pauli_vecs = pauli_complete_algebra_inner(old_Ops, start, n, old_basis, old_pauli_vecs)

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


def pauli_find_algebra(Op_0: list, Op_1: list, max: int):
    """
    Extend operators Op_0 to include commutations with operators in Op_1 and complete the algebra.

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
    if not isinstance(Op_0, list):
        raise TypeError(f"Op_0 must be a list, but {Op_0} is not.")
    for i, op in enumerate(Op_0):
        if not isinstance(op, Pauli) and not isinstance(op, SuperPauli):
            raise TypeError(f"Op_0 must be a list of Pauli tensor products (Pauli or SuperPauli), but {op} is not.")
        if isinstance(op, Pauli):
            Op_0[i] = SuperPauli([(1, op)])
    if not isinstance(Op_1, list):
        raise TypeError(f"Op_1 must be a list, but {Op_1} is not.")
    for i, op in enumerate(Op_1):
        if not isinstance(op, Pauli) and not isinstance(op, SuperPauli):
            raise TypeError(f"Op_1 must be a list of Pauli tensor products (Pauli or SuperPauli), but {op} is not.")
        if isinstance(op, Pauli):
            Op_1[i] = SuperPauli([(1, op)])

    Ops = Op_0.copy()
    ind, old_basis, old_pauli_vecs = lin_ind(Ops)
    n = len(Ops)
    if not ind:
        raise LinearIndependenceError(
            "Given operators in Op_0 are not linearly independent."
        )
    # append every commutation that is linearly independent
    for i in range(len(Op_0)):
        for j in range(len(Op_1)):
            new_op = comm(Op_0[i], Op_1[j])
            Ops.append(new_op)
            ind, old_basis, old_pauli_vecs = lin_ind(Ops, n, old_basis, old_pauli_vecs)
            if not ind:
                Ops.pop()
                old_pauli_vecs.pop()
            else:
                n += 1
    # complete the algebra
    Lie_alg = pauli_complete_algebra(Ops, max)
    return Lie_alg

def tensor(Op: Pauli, dim: int, loc: list):
    """
    Calculate tensor product of Op acting on qubits in position loc and trivially elsewhere.
    
    Parameters
    ----------
    Op : SuperPauli or Pauli
        Non-trivial operator
    dim : int
        Number of qubits
    loc : list
        List of qubits to apply Op to

    Returns
    -------
    Pauli
        Tensor product acting with Op on qubits in loc and trivially elsewhere
    """
    if not (Op == I).all() and not (Op == X).all() and not (Op == Y).all() and not (Op == Z).all():
        raise ValueError(f"Op must be a Pauli matrix, but {Op} is not.")
    if not isinstance(dim, int):
        raise TypeError(f"dim must be an integer, but {dim} is not.")
    if not isinstance(loc, list):
        raise TypeError(f"loc must be a list, but {loc} is not.")
    for i in loc:
        if not isinstance(i, int):
            raise TypeError(f"loc must be a list of integers, but {i} is not.")

    prod_paulis = [I for i in range(dim)]
    for i in loc:
        prod_paulis[i] = Op
    return Pauli(prod_paulis)

def x_tensor(dim: int, loc: list):
    """
    Returns tensor(X, dim, loc).
    """
    return tensor(X, dim, loc)

def y_tensor(dim: int, loc: list):
    """
    Returns tensor(Y, dim, loc).
    """
    return tensor(Y, dim, loc)

def z_tensor(dim: int, loc: list):
    """
    Returns tensor(Z, dim, loc).
    """
    return tensor(Z, dim, loc)

def pauli_make_algebra(Op_1: list, Op_2: list, max: int):
    """
    Construct set of operators from Op_1 closed under commutations with Op_2.
    
    Parameters
    ----------
    Op_1 : list
        List of operators (defines invariants)
    Op_2 : list
        List of operators (defines H)
    max : int
        Stop iterations at maximum number of operators in algebra

    Returns
    -------
    Lie_alg : list
        List of operators in extended Lie algebra
    """
    if not isinstance(Op_1, list):
        raise TypeError(f"Op_1 must be a list, but {Op_1} is not.")
    for i, op in enumerate(Op_1):
        if not isinstance(op, Pauli) and not isinstance(op, SuperPauli):
            raise TypeError(f"Op_1 must be a list of Pauli tensor products (Pauli or SuperPauli), but {op} is not.")
        if isinstance(op, Pauli):
            Op_1[i] = SuperPauli([(1, op)])
    if not isinstance(Op_2, list):
        raise TypeError(f"Op_2 must be a list, but {Op_2} is not.")
    for i, op in enumerate(Op_2):
        if not isinstance(op, Pauli) and not isinstance(op, SuperPauli):
            raise TypeError(f"Op_2 must be a list of Pauli tensor products (Pauli or SuperPauli), but {op} is not.")
        if isinstance(op, Pauli):
            Op_2[i] = SuperPauli([(1, op)])

    op_1, op_2 = Op_1.copy(), Op_2.copy()
    Lie_alg = Op_1.copy()
    old_basis, old_pauli_vecs = [], []
    n = 0
    while True:
        for op in op_2:
            new_op = comm(op_1[0], op)
            if len(new_op) == 0:
                break
            Lie_alg.append(new_op)
            ind, old_basis, old_pauli_vecs = lin_ind(Lie_alg, n, old_basis, old_pauli_vecs)
            if not ind:
                Lie_alg.pop()
                old_pauli_vecs.pop()
            else:
                n = len(Lie_alg)
                op_1.append(new_op)
            if len(Lie_alg) > max:
                raise MaxOperatorsError(
                    f"Maximum of {max} operators in uncomplete algebra reached."
                )
        op_1.pop(0)
        if len(op_1) == 0:
            break
    return Lie_alg
