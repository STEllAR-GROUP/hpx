#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <cudaq.h>
#include <iostream>

@cudaq.kernel
def quantum_task():
    q = cudaq.qalloc(2)
    cudaq.h(q[0])
    cudaq.cx(q[0], q[1])

void execute_quantum_task() {
    std::cout << "Executing Quantum Task in HPX Executor" << std::endl;
    quantum_task();
}

int main() {
    auto future_task = hpx::async(&execute_quantum_task);
    future_task.get();
    std::cout << "Quantum task execution completed." << std::endl;
    return 0;
}
