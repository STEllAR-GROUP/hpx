//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_STEPPER_HPP
#define HPX_EXAMPLES_MINI_STEPPER_HPP

#include <examples/mini_ghost/params.hpp>
#include <examples/mini_ghost/global_sum.hpp>
#include <examples/mini_ghost/partition.hpp>
#include <examples/mini_ghost/send_buffer.hpp>
#include <examples/mini_ghost/spikes.hpp>
#include <examples/mini_ghost/recv_buffer.hpp>

#include <hpx/include/components.hpp>
#include <hpx/lcos/broadcast.hpp>

#include <boost/random.hpp>

namespace mini_ghost {
    template <typename Real>
    struct stepper
      : hpx::components::managed_component_base<stepper<Real> >
    {
        static const std::size_t max_num_neighbors = 6;

        typedef hpx::util::serialize_buffer<Real> buffer_type;

        stepper();

        hpx::future<void> init(params<Real> & p);

        void run(std::size_t num_spikes, std::size_t num_tsteps);

        ///////////////////////////////////////////////////////////////////////
        // define actions to set the boundary data
        void set_global_sum(std::size_t generation, std::size_t which,
            Real value, std::size_t idx, std::size_t id);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_global_sum,
            set_global_sum_action);

        void set_north_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_north_zone,
            set_north_zone_action);

        void set_south_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_south_zone,
            set_south_zone_action);

        void set_east_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_east_zone,
            set_east_zone_action);

        void set_west_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_west_zone,
            set_west_zone_action);

        void set_front_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_front_zone,
            set_front_zone_action);

        void set_back_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_back_zone,
            set_back_zone_action);

        std::size_t get_rank() const { return rank; }

    private:
        boost::random::mt19937 gen;
        boost::random::uniform_real_distribution<Real> random;

        std::size_t rank;
        std::vector<hpx::id_type> stepper_ids;

        std::vector<spikes<Real> > spikes_;

        typedef
            partition<
                Real
              , set_global_sum_action
              , set_south_zone_action, set_north_zone_action
              , set_west_zone_action, set_east_zone_action
              , set_front_zone_action, set_back_zone_action
            >
            partition_type;
        std::vector<partition_type> partitions_;

        hpx::lcos::local::promise<void> init_promise_;
        hpx::future<void> init_future_;
    };
}

HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION(
    mini_ghost::stepper<float>::set_global_sum_action
  , mini_ghost_stepper_float_set_global_sum_action
)

HPX_REGISTER_BROADCAST_APPLY_ACTION_DECLARATION(
    mini_ghost::stepper<double>::set_global_sum_action
  , mini_ghost_stepper_double_set_global_sum_action
)

#endif
