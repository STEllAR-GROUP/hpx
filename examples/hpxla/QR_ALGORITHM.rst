.. Copyright (c) 2012 Bryce Adelstein-Lelbach
..  
.. Distributed under the Boost Software License, Version 1.0. (See accompanying
.. file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

Notes:

* {} is used to indicate subscript.
* ||x|| means the euclidean norm of x.
* x^T means the transpose of x.
* I means the identity matrix.
 
Practical QR Algorithm
----------------------

Note: This is not the fastest QR eigenvalue algorithm. The implicit QR algorithm
is faster.

Let A be a real matrix.

    A{0} = A

    for (k = 0; not_converged; ++k)
        A{k+1} = R{k} Q{k}

Q{k} is an orthogonal matrix and R{k} is an upper triangular matrix such that:

    A{k} == Q{k} R{k}

The matrices A{k} will eventually converge to a triangular matrix with the same
eigenvalues as A. We use Householders transforms to compute Q{k} and R{k}.

Householders QR Factorization
-----------------------------

Let A be a real n x n matrix.

    R{0} = A
    Q{0} = I

    for (l = 0; l < (n-1); ++l)
        s{l} = sign(R[l][l])
        sigma{l} = sqrt(R[l][l]^2 + R[l+1][l+1]^2 + ... + R[n][l]^2)

        w{l}[l] = R[l][l] + sign{l} * sigma{l}
        
        for (i = (l+1); i < n; ++i)
            w{l}[i] = R[i][l]

        v{l} = w{l} / ||w{l}||

        H{l} = I - 2 * v * v^T

        R{l+1} = H{l} * R{l}
        Q{l+1} = Q{l} * H{l}

Q{k} and R{k} will converge in n - 1 iterations to a QR decomposition of A.


