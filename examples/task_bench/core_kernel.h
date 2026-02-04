/* Copyright 2020 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef CORE_KERNEL_H
#define CORE_KERNEL_H

#include <cstddef>

struct Kernel;

void execute_kernel_empty(const Kernel &kernel);

long long execute_kernel_busy_wait(const Kernel &kernel);

void execute_kernel_memory(const Kernel &kernel,
                           char *scratch_large_ptr, size_t scratch_large_bytes, 
                           long timestep);

void execute_kernel_dgemm(const Kernel &kernel,
                          char *scratch_ptr, size_t scratch_bytes);

void execute_kernel_daxpy(const Kernel &kernel,
                          char *scratch_large_ptr, size_t scratch_large_bytes, 
                          long timestep);

double execute_kernel_compute(const Kernel &kernel);

double execute_kernel_compute2(const Kernel &kernel);

void execute_kernel_io(const Kernel &kernel);

long select_imbalance_iterations(const Kernel &kernel,
                                 long graph_index, long timestep, long point);

double execute_kernel_imbalance(const Kernel &kernel,
                                long graph_index, long timestep, long point);

#endif
