// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(EXAMPLES_SGAB_BC_SSSP_20110217T0827)
#define EXAMPLES_SGAB_BC_SSSP_20110217T0827

#include <algorithm>
#include <queue>

// Bring in PXGL headers
#include <pxgl/pxgl.hpp>

#include <pxgl/xua/range.hpp>
#include <pxgl/xua/arbitrary_distribution.hpp>

#include <pxgl/graphs/csr_graph.hpp>
#include <pxgl/graphs/edge_tuple.hpp>
#include <pxgl/xua/vector.hpp>
#include <pxgl/xua/numeric.hpp>

#include <pxgl/util/component.hpp>

////////////////////////////////////////////////////////////////////////////////
namespace examples {
  namespace sgab {
    class bc_sssp;
  }
}

////////////////////////////////////////////////////////////////////////////////
namespace examples { namespace sgab { namespace server {

  class bc_sssp
    : public HPX_MANAGED_BASE_0(bc_sssp)
  {
  public:
    enum actions
    {
      // Construction/initialization
      bc_sssp_instantiate,
      bc_sssp_replicate,
      bc_sssp_constructed,
      bc_sssp_ready,
      bc_sssp_ready_all,
      bc_sssp_ended,
      // Use
      bc_sssp_begin,
      bc_sssp_expand_source,
      bc_sssp_expand_target,
      bc_sssp_contract_target,
      bc_sssp_contract_source,
    };

    ////////////////////////////////////////////////////////////////////////////
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef hpx::naming::gid_type gid_type;
    typedef std::vector<hpx::naming::gid_type> gids_type;

    typedef unsigned long size_type;
    typedef std::vector<size_type> sizes_type;

    typedef std::vector<long> longs_type;

    typedef std::vector<sizes_type> predecessors_type;

    typedef std::vector<double> doubles_type;

    typedef pxgl::xua::arbitrary_distribution<id_type, pxgl::xua::range>
            arbitrary_distribution_type;
    typedef arbitrary_distribution_type distribution_type;

    typedef pxgl::xua::vector<
        arbitrary_distribution_type,
        pxgl::graphs::server::edge_tuple_type
    > edge_container_client_type;

    typedef pxgl::graphs::csr_graph<
        edge_container_client_type, 
        arbitrary_distribution_type
    > graph_type;

    typedef pxgl::xua::numeric<
        arbitrary_distribution_type,
        double
    > bc_scores_type;
    typedef bc_scores_type::server_type bc_scores_member_type;

    typedef examples::sgab::bc_sssp bc_sssp_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction/initialization
    bc_sssp();
    ~bc_sssp();

  private:
    inline void initialize_local_variables(void);
    inline size_type const index(size_type const);

  public:
    void instantiate(
        id_type const & me,
        distribution_type const & distribution,
        id_type const & graph_id,
        id_type const & bc_scores_id);
    typedef hpx::actions::action4<
        bc_sssp, 
        bc_sssp_instantiate, 
            id_type const &, 
            distribution_type const &, 
            id_type const &,
            id_type const &,
        &bc_sssp::instantiate
    > instantiate_action;

    void replicate(
        distribution_type const & distribution, 
        ids_type const & sibling_ids,
        id_type const & graph_id,
        id_type const & bc_scores_id);
    typedef hpx::actions::action4<
        bc_sssp, 
        bc_sssp_replicate, 
            distribution_type const &, 
            ids_type const &,
            id_type const &,
            id_type const &,
        &bc_sssp::replicate
    > replicate_action;

    void constructed(void);
    typedef hpx::actions::action0<
        bc_sssp, 
        bc_sssp_constructed, 
        &bc_sssp::constructed
    > constructed_action;

    void not_constructed(void);

    void ready(void);
    typedef hpx::actions::action0<
        bc_sssp, 
        bc_sssp_ready, 
        &bc_sssp::ready
    > ready_action;

    void not_ready(void);

    void ready_all(void);
    typedef hpx::actions::action0<
        bc_sssp, 
        bc_sssp_ready_all, 
        &bc_sssp::ready_all
    > ready_all_action;

    void ended(void);
    typedef hpx::actions::action0<
        bc_sssp, 
        bc_sssp_ended, 
        &bc_sssp::ended
    > ended_action;

    ////////////////////////////////////////////////////////////////////////////
    // Use interface
    void begin(size_type);
    typedef hpx::actions::action1<
        bc_sssp, 
        bc_sssp_begin, 
            size_type,
        &bc_sssp::begin
    > begin_action;

    sizes_type expand_source(size_type);
    typedef hpx::actions::result_action1<
        bc_sssp, 
            sizes_type,
        bc_sssp_expand_source, 
            size_type,
        &bc_sssp::expand_source
    > expand_source_action;

    size_type expand_target(
        size_type,
        size_type,
        long,
        size_type);
    typedef hpx::actions::result_action4<
        bc_sssp, 
            size_type,
        bc_sssp_expand_target, 
            size_type,
            size_type,
            long,
            size_type,
        &bc_sssp::expand_target
    > expand_target_action;

    void contract_target(
        size_type,
        size_type);
    typedef hpx::actions::action2<
        bc_sssp, 
        bc_sssp_contract_target, 
            size_type,
            size_type,
        &bc_sssp::contract_target
    > contract_target_action;

    void contract_source(
        size_type,
        size_type,
        size_type,
        double);
    typedef hpx::actions::action4<
        bc_sssp, 
        bc_sssp_contract_source, 
            size_type,
            size_type,
            size_type,
            double,
        &bc_sssp::contract_source
    > contract_source_action;

    ////////////////////////////////////////////////////////////////////////////
  private:
    id_type me_;
    id_type here_;

    distribution_type distribution_;

    std::vector<bc_sssp_type> siblings_;

    ////////////////////////////////////////////////////////////////////////////
    // Local variables
    graph_type graph_;
    bc_scores_type bc_scores_;

    doubles_type * BC_ptr_;

    std::queue<size_type> Q_;
    sizes_type S_;

    predecessors_type P_;
    longs_type d_;
    sizes_type sigma_;
    doubles_type delta_;

    ////////////////////////////////////////////////////////////////////////////
    // Synchronization members
    struct tag {};
    typedef hpx::util::spinlock_pool<tag> mutex_type;
    typedef mutex_type::scoped_lock scoped_lock;

    typedef int result_type;
    typedef boost::exception_ptr error_type;
    typedef boost::variant<result_type, error_type> feb_data_type;

    // Used to suspend calling threads until process is constructed
    // Note: this is required because we cannot pass arguments to the
    // component constructor
    bool constructed_;
    hpx::util::full_empty<feb_data_type> constructed_feb_;

    // Used to suspend calling threads until process is initialized
    bool initialized_;
    hpx::util::full_empty<feb_data_type> initialized_feb_;

    // Used to suspend calling threads until process is ended
    bool ended_;
    hpx::util::full_empty<feb_data_type> ended_feb_;

    // Use to block threads around critical sections
    hpx::util::full_empty<feb_data_type> use_feb_;
  };
}}}

#endif
