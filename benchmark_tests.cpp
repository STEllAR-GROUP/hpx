#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <chrono>
#include <iostream>

void timed_task(int task_id) {
    auto start = std::chrono::high_resolution_clock::now();
    hpx::this_thread::sleep_for(std::chrono::seconds(2));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Task " << task_id << " completed in " << elapsed.count() << " seconds." << std::endl;
}

int main() {
    const int num_tasks = 5;
    std::vector<hpx::future<void>> tasks;
    for (int i = 0; i < num_tasks; i++) {
        tasks.push_back(hpx::async(&timed_task, i));
    }
    hpx::wait_all(tasks);
    std::cout << "Benchmarking completed." << std::endl;
    return 0;
}
