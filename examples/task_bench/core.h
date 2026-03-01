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
#ifndef CORE_H
#define CORE_H

#include "core_c.h"

#include <string>
#include <vector>

typedef dependence_type_t DependenceType;

typedef kernel_type_t KernelType;

struct TaskGraph;

struct Kernel : public kernel_t
{
    Kernel() = default;
    Kernel(kernel_t k)
      : kernel_t(k)
    {
    }

private:
    void execute(long graph_index, long timestep, long point, char* scratch_ptr,
        size_t scratch_bytes) const;
    friend struct TaskGraph;
};

struct TaskGraph : public task_graph_t
{
    TaskGraph() = default;
    TaskGraph(task_graph_t t)
      : task_graph_t(t)
    {
    }

    long offset_at_timestep(long timestep) const;
    long width_at_timestep(long timestep) const;

    long max_dependence_sets() const;

    long timestep_period() const;
    long dependence_set_at_timestep(long timestep) const;

    std::vector<std::pair<long, long>> reverse_dependencies(
        long dset, long point) const;
    std::vector<std::pair<long, long>> dependencies(
        long dset, long point) const;

    size_t reverse_dependencies(
        long dset, long point, std::pair<long, long>* deps) const;
    size_t dependencies(
        long dset, long point, std::pair<long, long>* deps) const;

    size_t num_reverse_dependencies(long dset, long point) const;
    size_t num_dependencies(long dset, long point) const;

    void execute_point(long timestep, long point, char* output_ptr,
        size_t output_bytes, char const** input_ptr, size_t const* input_bytes,
        size_t n_inputs, char* scratch_ptr, size_t scratch_bytes) const;
    static void prepare_scratch(char* scratch_ptr, size_t scratch_bytes);
};

struct App
{
    std::vector<TaskGraph> graphs;
    long nodes;
    int verbose;
    bool enable_graph_validation;

    App(int argc, char** argv);
    void check() const;
    void display() const;
    void report_timing(double elapsed_seconds) const;
};

static_assert(std::is_pod<Kernel>::value, "Kernel must be POD");
static_assert(std::is_pod<TaskGraph>::value, "TaskGraph must be POD");

long long count_flops_per_task(TaskGraph const& g, long timestep, long point);
long long count_bytes_per_task(TaskGraph const& g, long timestep, long point);

#endif
