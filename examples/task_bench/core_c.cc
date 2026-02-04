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
#include "core.h"
#include "core_c.h"

interval_list_t wrap_consume(std::vector<std::pair<long, long> > &&d) {
  std::vector<std::pair<long, long> > *d_ptr = new std::vector<std::pair<long, long> >;
  d_ptr->swap(d);
  interval_list_t result;
  result.impl = reinterpret_cast<void *>(d_ptr);
  return result;
}

std::vector<std::pair<long, long> > * unwrap(interval_list_t d) {
  return reinterpret_cast<std::vector<std::pair<long, long> > *>(d.impl);
}

interval_t wrap(const std::pair<long, long> &p) {
  interval_t result;
  result.start = p.first;
  result.end = p.second;
  return result;
}

task_graph_list_t wrap(const std::vector<TaskGraph> &g) {
  std::vector<TaskGraph> *g_ptr = new std::vector<TaskGraph>(g);
  task_graph_list_t result;
  result.impl = reinterpret_cast<void *>(g_ptr);
  return result;
}

std::vector<TaskGraph> * unwrap(task_graph_list_t a) {
  return reinterpret_cast<std::vector<TaskGraph> *>(a.impl);
}

app_t wrap(App *a_ptr) {
  app_t result;
  result.impl = reinterpret_cast<void *>(a_ptr);
  return result;
}

App * unwrap(app_t a) {
  return reinterpret_cast<App *>(a.impl);
}

long task_graph_offset_at_timestep(task_graph_t graph, long timestep)
{
  TaskGraph t(graph);
  return t.offset_at_timestep(timestep);
}

long task_graph_width_at_timestep(task_graph_t graph, long timestep)
{
  TaskGraph t(graph);
  return t.width_at_timestep(timestep);
}

long task_graph_max_dependence_sets(task_graph_t graph)
{
  TaskGraph t(graph);
  return t.max_dependence_sets();
}

long task_graph_timestep_period(task_graph_t graph)
{
  TaskGraph t(graph);
  return t.timestep_period();
}

long task_graph_dependence_set_at_timestep(task_graph_t graph, long timestep)
{
  TaskGraph t(graph);
  return t.dependence_set_at_timestep(timestep);
}

interval_list_t task_graph_reverse_dependencies(task_graph_t graph, long dset, long point)
{
  TaskGraph t(graph);
  return wrap_consume(t.reverse_dependencies(dset, point));
}

interval_list_t task_graph_dependencies(task_graph_t graph, long dset, long point)
{
  TaskGraph t(graph);
  return wrap_consume(t.dependencies(dset, point));
}

void task_graph_execute_point_scratch(task_graph_t graph, long timestep, long point,
                                      char *output_ptr, size_t output_bytes,
                                      const char **input_ptr, const size_t *input_bytes,
                                      size_t n_inputs,
                                      char *scratch_ptr, size_t scratch_bytes)
{
  TaskGraph t(graph);
  t.execute_point(timestep, point, output_ptr, output_bytes,
                  input_ptr, input_bytes, n_inputs,
                  scratch_ptr, scratch_bytes);
}

void task_graph_execute_point_scratch_auto(task_graph_t graph, long timestep, long point,
                                           char *output_ptr, size_t output_bytes,
                                           const char **input_ptr, const size_t *input_bytes,
                                           size_t n_inputs,
                                           size_t scratch_bytes)
{
  std::vector<char> scratch(scratch_bytes);
  TaskGraph::prepare_scratch(scratch.data(), scratch.size());
  TaskGraph t(graph);
  t.execute_point(timestep, point, output_ptr, output_bytes,
                  input_ptr, input_bytes, n_inputs,
                  const_cast<char *>(scratch.data()), scratch.size());
}

void task_graph_execute_point_nonconst(task_graph_t graph, long timestep, long point,
                                       int64_t *output_ptr, size_t output_bytes,
                                       int64_t **input_ptr, const size_t *input_bytes,
                                       size_t n_inputs)
{
  TaskGraph t(graph);
  t.execute_point(timestep, point, reinterpret_cast<char *>(output_ptr), output_bytes,
                  reinterpret_cast<const char **>(const_cast<const int64_t **>(input_ptr)),
                  input_bytes, n_inputs,
                  NULL, 0);
}

void task_graph_execute_point_scratch_nonconst(task_graph_t graph, long timestep, long point,
                                               int64_t *output_ptr, size_t output_bytes,
                                               int64_t **input_ptr, const size_t *input_bytes,
                                               size_t n_inputs,
                                               char *scratch_ptr, size_t scratch_bytes)
{
  TaskGraph t(graph);
  t.execute_point(timestep, point, reinterpret_cast<char *>(output_ptr), output_bytes,
                  reinterpret_cast<const char **>(const_cast<const int64_t **>(input_ptr)),
                  input_bytes, n_inputs,
                  scratch_ptr, scratch_bytes);
}

void task_graph_prepare_scratch(char *scratch_ptr, size_t scratch_bytes)
{
  TaskGraph::prepare_scratch(scratch_ptr, scratch_bytes);
}

void interval_list_destroy(interval_list_t intervals)
{
  std::vector<std::pair<long, long> > *i = unwrap(intervals);
  delete i;
}

long interval_list_num_intervals(interval_list_t intervals)
{
  std::vector<std::pair<long, long> > *i = unwrap(intervals);
  return i->size();
}

interval_t interval_list_interval(interval_list_t intervals, long index)
{
  std::vector<std::pair<long, long> > *i = unwrap(intervals);
  return wrap((*i)[index]);
}

void task_graph_list_destroy(task_graph_list_t graphs)
{
  std::vector<TaskGraph> *g = unwrap(graphs);
  delete g;
}

long task_graph_list_num_task_graphs(task_graph_list_t graphs)
{
  std::vector<TaskGraph> *g = unwrap(graphs);
  return g->size();
}

task_graph_t task_graph_list_task_graph(task_graph_list_t graphs, long index)
{
  std::vector<TaskGraph> *g = unwrap(graphs);
  return (*g)[index];
}

app_t app_create(int argc, char **argv)
{
  App *app = new App(argc, argv);
  return wrap(app);
}

void app_destroy(app_t app)
{
  App *a = unwrap(app);
  delete a;
}

task_graph_list_t app_task_graphs(app_t app)
{
  App *a = unwrap(app);
  return wrap(a->graphs);
}

bool app_verbose(app_t app)
{
  App *a = unwrap(app);
  return a->verbose;
}

void app_check(app_t app)
{
  App *a = unwrap(app);
  a->check();
}

void app_display(app_t app)
{
  App *a = unwrap(app);
  a->display();
}

void app_report_timing(app_t app, double elapsed_seconds)
{
  App *a = unwrap(app);
  a->report_timing(elapsed_seconds);
}
