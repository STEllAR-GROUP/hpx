// File: hpx_scheduler.cpp (Week 3 - HPX Task Scheduling)
#include <hpx/hpx_main.hpp>
#include <hpx/include/async.hpp>
#include <iostream>
#include <vector>

void execute_task(int task_id) {
    std::cout << "Executing Task " << task_id 
              << " on thread: " << hpx::get_worker_thread_num() << std::endl;
    hpx::this_thread::sleep_for(std::chrono::seconds(2));
}

int main() {
    const int num_tasks = 5;
    std::vector<hpx::future<void>> tasks;
    for (int i = 0; i < num_tasks; i++) {
        tasks.push_back(hpx::async(&execute_task, i));
    }
    hpx::wait_all(tasks);
    std::cout << "All tasks completed." << std::endl;
    return 0;
}
