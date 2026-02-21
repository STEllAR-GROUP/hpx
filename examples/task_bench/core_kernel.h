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

void execute_kernel_empty(Kernel const& kernel);

long long execute_kernel_busy_wait(Kernel const& kernel);

void execute_kernel_memory(Kernel const& kernel, char* scratch_large_ptr,
    size_t scratch_large_bytes, long timestep);

void execute_kernel_dgemm(
    Kernel const& kernel, char* scratch_ptr, size_t scratch_bytes);

void execute_kernel_daxpy(Kernel const& kernel, char* scratch_large_ptr,
    size_t scratch_large_bytes, long timestep);

double execute_kernel_compute(Kernel const& kernel);

double execute_kernel_compute2(Kernel const& kernel);

void execute_kernel_io(Kernel const& kernel);

long select_imbalance_iterations(
    Kernel const& kernel, long graph_index, long timestep, long point);

double execute_kernel_imbalance(
    Kernel const& kernel, long graph_index, long timestep, long point);

#endif
