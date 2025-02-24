// File: task_executor.cpp (Week 2 - C++ Task Execution)
#include <iostream>
#include <thread>
#include <chrono>

void submit_task(std::string task_name) {
    std::cout << "Task '" << task_name << "' is being submitted." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::cout << "Task '" << task_name << "' is complete." << std::endl;
}

int main() {
    std::string tasks[] = {"Task 1", "Task 2", "Task 3"};
    for (const auto& task : tasks) {
        submit_task(task);
    }
    return 0;
}
