//  Copyright (c) 2009-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLICATIONS_SSCA2_BENCHMARK_SEP_22_2009_0228PM)
#define HPX_APPLICATIONS_SSCA2_BENCHMARK_SEP_22_2009_0228PM

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
// Helper routines

inline hpx::naming::id_type find_here(void)
{
    return hpx::applier::get_applier().get_runtime_support_gid();
}

///////////////////////////////////////////////////////////////////////////////
// Definitions for Kernel 1

int kernel1(naming::id_type G, std::string filename);

typedef actions::plain_result_action2<int,
                                      naming::id_type, std::string,
                                      kernel1>
kernel1_action;

///////////////////////////////////////////////////////////////////////////////
// Definitions for Kernel 2

int kernel2(naming::id_type G, naming::id_type dist_edge_set);
int large_set_local(naming::id_type local_vertex_set,
                             naming::id_type edge_set,
                             naming::id_type local_max_lco,
                             naming::id_type global_max_lco);

typedef
    actions::plain_result_action2<int, naming::id_type, naming::id_type, kernel2>
kernel2_action;

typedef
    actions::plain_result_action4<int, naming::id_type, naming::id_type,
                                  naming::id_type, naming::id_type,
                                  large_set_local>
large_set_local_action;

///////////////////////////////////////////////////////////////////////////////
// Definitions for Kernel 3

int kernel3(naming::id_type edge_set, naming::id_type subgraphs);
int extract_local(naming::id_type local_edge_set, naming::id_type local_subgraphs);
naming::id_type extract_subgraph(naming::id_type H, naming::id_type pmap,
                        naming::id_type source, naming::id_type target,
                        int d);

typedef
    actions::plain_result_action2<int, naming::id_type, naming::id_type, kernel3>
kernel3_action;

typedef
    actions::plain_result_action2<int, naming::id_type, naming::id_type, extract_local>
extract_local_action;

typedef
    actions::plain_result_action5<naming::id_type,
                                  naming::id_type, naming::id_type, naming::id_type,
                                  naming::id_type, int,
                                  extract_subgraph>
extract_subgraph_action;

///////////////////////////////////////////////////////////////////////////////
// Definitions for Kernel 4

int kernel4(naming::id_type G, naming::id_type VS, int k4_approx, naming::id_type bc_scores);
void select_random_vertices(std::vector<naming::id_type> v_locals, int k4_approx, naming::id_type VS);
double calculate_teps(naming::id_type V, int order, double total_time, bool exact);
int init_local_bc(naming::id_type bc_local, naming::id_type v_local);
int add_local_item(int index, naming::id_type v_locals, naming::id_type vs_locals);
int bfs_sssp_local(naming::id_type V, naming::id_type vs_locals, naming::id_type bc_locals);
int bfs_sssp(naming::id_type start, naming::id_type V, naming::id_type bc_locals);
int incr_bc(naming::id_type bc_scores, naming::id_type w, double delta_w);

typedef
    actions::plain_result_action4<int, naming::id_type, naming::id_type, int, naming::id_type, kernel4>
kernel4_action;

typedef
    actions::plain_result_action2<int, naming::id_type, naming::id_type, init_local_bc>
init_local_bc_action;

typedef
    actions::plain_result_action3<int, int, naming::id_type, naming::id_type, add_local_item>
add_local_item_action;

typedef
    actions::plain_result_action3<int, naming::id_type, naming::id_type, naming::id_type, bfs_sssp_local>
bfs_sssp_local_action;

typedef
    actions::plain_result_action3<int, naming::id_type, naming::id_type, naming::id_type, bfs_sssp>
bfs_sssp_action;

typedef
    actions::plain_result_action3<int, naming::id_type, naming::id_type, double, incr_bc>
incr_bc_action;

#endif
