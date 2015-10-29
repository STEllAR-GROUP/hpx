//  Copyright (c) 2011 Matthew Anderson
//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXAMPLE_SERVER_ALLGATHER_AND_GATE_JUL_13_2012_0147PM)
#define HPX_EXAMPLE_SERVER_ALLGATHER_AND_GATE_JUL_13_2012_0147PM

#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/include/util.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace ag { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT allgather_and_gate
      : public hpx::components::component_base<allgather_and_gate>
    {
    private:
        typedef hpx::lcos::local::spinlock mutex_type;

    public:
        allgather_and_gate()
        {}

        ///////////////////////////////////////////////////////////////////////
        void init(std::vector<hpx::naming::id_type> const& components,
            std::size_t rank);
        void compute(int num_loops);
        void print();

        HPX_DEFINE_COMPONENT_ACTION(allgather_and_gate, init, init_action);
        HPX_DEFINE_COMPONENT_ACTION(allgather_and_gate, compute, compute_action);
        HPX_DEFINE_COMPONENT_ACTION(allgather_and_gate, print, print_action);

        ///////////////////////////////////////////////////////////////////////
        // helper action for all-gather operation
        void set_data(std::size_t which, std::size_t generation, double data);
        HPX_DEFINE_COMPONENT_ACTION(allgather_and_gate, set_data,set_data_action);

    protected:
        void allgather(double value);

    private:
        mutable mutex_type mtx_;

        std::size_t rank_;                              // our rank

        std::vector<double> n_;                         // all-gathered values
        std::vector<hpx::naming::id_type> components_;  // components for allgather

        hpx::lcos::local::base_and_gate<> gate_;        // synchronization gate
    };
}}

// Declaration of serialization support for the actions
HPX_REGISTER_ACTION_DECLARATION(
    ag::server::allgather_and_gate::compute_action,
    allgather_compute_action);

HPX_REGISTER_ACTION_DECLARATION(
    ag::server::allgather_and_gate::print_action,
    allgather_print_action);

HPX_REGISTER_ACTION_DECLARATION(
    ag::server::allgather_and_gate::set_data_action,
    allgather_set_data_action);

#endif

