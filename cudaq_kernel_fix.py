import cudaq

@cudaq.kernel
def corrected_kernel():
    q = cudaq.qalloc(3)  # Allocate 3 qubits
    cudaq.x(q[0])  # Apply X gate correctly
    cudaq.h(q[1])  # Apply H gate to a valid index

corrected_kernel()
