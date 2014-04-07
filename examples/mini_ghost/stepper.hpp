//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_STEPPER_HPP
#define HPX_EXAMPLES_MINI_STEPPER_HPP

#include <examples/mini_ghost/params.hpp>
#include <examples/mini_ghost/global_sum.hpp>
#include <examples/mini_ghost/grid.hpp>
#include <examples/mini_ghost/send_buffer.hpp>
#include <examples/mini_ghost/recv_buffer.hpp>

#include <hpx/include/components.hpp>

#include <random>

namespace mini_ghost {
    template <typename Real>
    struct stepper
      : hpx::components::managed_component_base<stepper<Real> >
    {
        static const std::size_t max_num_neighbors = 6;

        typedef
            hpx::util::serialize_buffer<Real>
            buffer_type;

        stepper();

        void init(params<Real> & p);

        void run(std::size_t num_spikes, std::size_t num_tsteps);

        void set_global_sum(std::size_t idx, std::size_t generation, std::size_t which, Real value);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_global_sum, set_global_sum_action);


        void set_north_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_north_zone, set_north_zone_action);

        void set_south_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_south_zone, set_south_zone_action);

        void set_east_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_east_zone, set_east_zone_action);

        void set_west_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_west_zone, set_west_zone_action);

        void set_front_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_front_zone, set_front_zone_action);

        void set_back_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_back_zone, set_back_zone_action);

        typedef
            send_buffer<
                buffer_type
              , NORTH
              , set_south_zone_action
            >
            send_buffer_north;

        typedef
            recv_buffer<
                buffer_type
              , NORTH
            >
            recv_buffer_north;

        typedef
            send_buffer<
                buffer_type
              , SOUTH
              , set_north_zone_action
            >
            send_buffer_south;

        typedef
            recv_buffer<
                buffer_type
              , SOUTH
            >
            recv_buffer_south;

        typedef
            send_buffer<
                buffer_type
              , EAST
              , set_west_zone_action
            >
            send_buffer_east;

        typedef
            recv_buffer<
                buffer_type
              , EAST
            >
            recv_buffer_east;

        typedef
            send_buffer<
                buffer_type
              , WEST
              , set_east_zone_action
            >
            send_buffer_west;

        typedef
            recv_buffer<
                buffer_type
              , WEST
            >
            recv_buffer_west;

        typedef
            send_buffer<
                buffer_type
              , FRONT
              , set_back_zone_action
            >
            send_buffer_front;

        typedef
            recv_buffer<
                buffer_type
              , FRONT
            >
            recv_buffer_front;

        typedef
            send_buffer<
                buffer_type
              , BACK
              , set_front_zone_action
            >
            send_buffer_back;

        typedef
            recv_buffer<
                buffer_type
              , BACK
            >
            recv_buffer_back;
    private:
        void setup_communication_parameter(params<Real> & p);
        void setup_global_indices(params<Real> & p);
        void setup_spikes(params<Real> & p);
        void setup_grids(params<Real> & p);

        void insert_spike(std::size_t spike);

        void flux_accumulate(std::size_t var);

        void print_header(params<Real> & p);

        std::mt19937 gen;
        std::uniform_real_distribution<Real> random;

        std::size_t rank;
        std::size_t num_neighs;
        std::vector<hpx::id_type> stepper_ids;

        std::vector<global_sum<Real> > global_sums;

        std::size_t nx;
        std::size_t ny;
        std::size_t nz;

        std::size_t npx;
        std::size_t npy;
        std::size_t npz;

        std::size_t my_px;
        std::size_t my_py;
        std::size_t my_pz;

        std::size_t global_nx;
        std::size_t global_ny;
        std::size_t global_nz;

        std::pair<std::size_t, std::size_t> my_global_nx;
        std::pair<std::size_t, std::size_t> my_global_ny;
        std::pair<std::size_t, std::size_t> my_global_nz;

        std::vector<Real> source_total;

        grid<Real> spikes;
        grid<std::size_t> spike_loc;

        std::size_t stencil;
        std::size_t num_vars;
        std::size_t report_diffusion;
        Real error_tol;

        std::vector<Real> flux_out;

        std::vector<send_buffer_north> send_buffer_norths;
        std::vector<send_buffer_south> send_buffer_souths;
        std::vector<send_buffer_east>  send_buffer_easts ;
        std::vector<send_buffer_west>  send_buffer_wests ;
        std::vector<send_buffer_front> send_buffer_fronts;
        std::vector<send_buffer_back>  send_buffer_backs ;

        std::vector<recv_buffer_north> recv_buffer_norths;
        std::vector<recv_buffer_south> recv_buffer_souths;
        std::vector<recv_buffer_east>  recv_buffer_easts ;
        std::vector<recv_buffer_west>  recv_buffer_wests ;
        std::vector<recv_buffer_front> recv_buffer_fronts;
        std::vector<recv_buffer_back>  recv_buffer_backs ;

        std::size_t src;
        std::size_t dst;
        std::array<std::vector<grid<Real>>, 2> grids;
        std::vector<bool> grids_to_sum;
        std::size_t num_sum_grid;
    };
}

#endif
