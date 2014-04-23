//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <examples/mini_ghost/profiling.hpp>
#include <examples/mini_ghost/stepper.hpp>
#include <examples/mini_ghost/stencils.hpp>
#include <examples/mini_ghost/write_grid.hpp>

#include <hpx/lcos/wait_all.hpp>

typedef hpx::components::managed_component<
    mini_ghost::stepper<float>
> stepper_float_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(stepper_float_type, stepper_float);

typedef hpx::components::managed_component<
    mini_ghost::stepper<double>
> stepper_double_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(stepper_double_type, stepper_double);

HPX_REGISTER_BROADCAST_APPLY_ACTION(
    mini_ghost::stepper<float>::set_global_sum_action
  , mini_ghost_stepper_float_set_global_sum_action
)

HPX_REGISTER_BROADCAST_APPLY_ACTION(
    mini_ghost::stepper<double>::set_global_sum_action
  , mini_ghost_stepper_double_set_global_sum_action
)

namespace mini_ghost {

    template <typename Real>
    stepper<Real>::stepper()
      : gen(0)
      , random(0.0, 1.0)
      , rank(-1)
      , init_future_(init_promise_.get_future())
    {}

    template <typename Real>
    void stepper<Real>::init(params<Real> & p)
    {
        // register our own name
        const std::string base_name = "/mini_ghost/stepper/";
        std::string name = base_name;
        name += boost::lexical_cast<std::string>(p.rank);

        std::vector<std::size_t> id_map(p.nranks);
        stepper_ids.resize(p.nranks);
        stepper_ids[0] = this->get_gid();
        stepper_ids[0].make_unmanaged();
        
        hpx::agas::register_name_sync(
            name
          , hpx::naming::detail::strip_credits_from_gid(stepper_ids[0].get_gid())
        );
        
        // Find all others ...
        std::size_t r = p.rank % p.nranks;
        for(std::size_t i = 0; i != p.nranks; ++i)
        {
            name = base_name;
            name += boost::lexical_cast<std::string>(r);
            std::size_t k = 0;

            while(!stepper_ids[i])
            {
                if(hpx::agas::resolve_name_sync(name, stepper_ids[i]))
                    break;
                hpx::lcos::local::spinlock::yield(k);
                ++k;
            }
            id_map[r] = i;
            r = (r + 1) % p.nranks;
        }

        rank = p.rank;

        // Set position in 3D processor grid
        std::size_t myrank_xy = p.rank % (p.npx*p.npy);
        std::size_t remainder = myrank_xy % p.npx;
        std::size_t my_py = myrank_xy / p.npx;
        std::size_t my_px = 0;
        if(remainder != 0)
        {
            my_px = remainder;
        }
        else
        {
            my_px = 0;
        }
        std::size_t my_pz = p.rank / (p.npx*p.npy);

        // Initialize grids and spikes
        std::vector<hpx::future<void> > partition_init_futures;//
        partition_init_futures.reserve(p.num_vars);

        auto random_lambda = [this]() -> Real { return random(gen); };
        std::size_t num_sum_grid = 0;
        partitions_.reserve(p.num_vars);
        for(std::size_t var = 0; var != p.num_vars; ++var)
        {
            spikes_.push_back(
                spikes<Real>(
                    p
                  , my_px
                  , my_py
                  , my_pz
                  , random_lambda
                )
            );
            partition_init_futures.push_back(hpx::future<void>());
            partitions_.push_back(
                partition_type(
                    var
                  , p
                  , stepper_ids
                  , id_map
                  , my_px
                  , my_py
                  , my_pz
                  , random_lambda
                  , partition_init_futures[var]
                )
            );
            if(partitions_.back().sum_grid()) ++num_sum_grid;
        }
        init_promise_.set_value();
        hpx::wait_all(partition_init_futures);
        //FIXME: setup performance captures...

        if(p.rank == 0)
        {
            p.print_header(num_sum_grid);
        }
    }

    template <typename Real>
    void stepper<Real>::run(std::size_t num_spikes, std::size_t num_tsteps)
    {
        hpx::util::high_resolution_timer timer_all;

        hpx::lcos::local::spinlock io_mutex;

        for(std::size_t spike = 0; spike != num_spikes; ++spike)
        {
            std::vector<hpx::future<void>> run_futures;
            run_futures.reserve(partitions_.size());
            for(auto & partition : partitions_)
            {
                partition.insert_spike(spikes_[spike], spike);
                run_futures.push_back(
                    hpx::async(
                        hpx::util::bind(
                            &partition_type::run
                          , boost::ref(partition)
                          , boost::ref(stepper_ids)
                          , num_tsteps
                          , boost::ref(io_mutex)
                        )
                /*
                        &partition_type::run
                      , boost::ref(partition)
                      , boost::ref(stepper_ids)
                      , num_tsteps
                      , boost::ref(io_mutex)
                */
                    )
                );
            }
            hpx::wait_all(run_futures);
        }
        profiling::data().time_wall(timer_all.elapsed());
    }

    template <typename Real>
    void stepper<Real>::set_north_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        init_future_.wait();
        partitions_[var].recv_buffer_north_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_south_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        init_future_.wait();
        partitions_[var].recv_buffer_south_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_east_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        init_future_.wait();
        partitions_[var].recv_buffer_east_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_west_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        init_future_.wait();
        partitions_[var].recv_buffer_west_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_front_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        init_future_.wait();
        partitions_[var].recv_buffer_front_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_back_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        init_future_.wait();
        partitions_[var].recv_buffer_back_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_global_sum(std::size_t generation, std::size_t which, Real value, std::size_t idx, std::size_t id)
    {
        init_future_.wait();
        HPX_ASSERT(which < stepper_ids.size());
        partitions_[idx].sum_allreduce(id, generation, which, value);
    }

    template struct stepper<float>;
    template struct stepper<double>;
}
