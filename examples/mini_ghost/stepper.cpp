//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <examples/mini_ghost/flux_accumulate.hpp>
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

namespace mini_ghost {

    template <typename Real>
    stepper<Real>::stepper()
      : gen(0)
      , random(0.0, 1.0)
      , rank(-1)
    {}

    template <typename Real>
    void stepper<Real>::init(params<Real> & p)
    {
        // register our own name
        const std::string base_name = "/mini_ghost/stepper/";
        std::string name = base_name;
        name += boost::lexical_cast<std::string>(p.rank);
        hpx::agas::register_name(name, this->get_gid());
        stepper_ids.resize(p.nranks);

        // Find all others ...
        stepper_ids[p.rank] = this->get_gid();
        for(std::size_t rank = 0; rank != p.nranks; ++rank)
        {
            name = base_name;
            name += boost::lexical_cast<std::string>(rank);
            while(!stepper_ids[rank])
            {
                stepper_ids[rank] = hpx::agas::resolve_name_sync(name);
            }
        }

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
        std::vector<hpx::future<void> > partition_init_futures(p.num_vars);
        auto random_lambda = [this]() -> Real { return random(gen); };
        std::size_t num_sum_grid = 0;
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
            partitions_.push_back(
                partition_type(
                    var
                  , p
                  , stepper_ids
                  , my_px
                  , my_py
                  , my_pz
                  , random_lambda
                  , partition_init_futures[var]
                )
            );
            if(partitions_.back().sum_grid()) ++num_sum_grid;
        }
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
                        &partition_type::run
                      , boost::ref(partition)
                      , boost::ref(stepper_ids)
                      , num_tsteps
                      , boost::ref(io_mutex)
                    )
                );
            }
            hpx::wait_all(run_futures);
        }
    }

    template <typename Real>
    void stepper<Real>::set_north_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        partitions_[var].recv_buffer_north_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_south_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        partitions_[var].recv_buffer_south_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_east_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        partitions_[var].recv_buffer_east_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_west_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        partitions_[var].recv_buffer_west_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_front_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        partitions_[var].recv_buffer_front_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_back_zone(buffer_type buffer, std::size_t step, std::size_t var)
    {
        partitions_[var].recv_buffer_back_.set_buffer(buffer, step);
    }

    template <typename Real>
    void stepper<Real>::set_global_sum(std::size_t generation, std::size_t which, Real value, std::size_t idx)
    {
        HPX_ASSERT(which < stepper_ids.size());
        partitions_[idx].sum_allreduce(generation, which, value);
    }

    /*
    template <typename Real>
    void stepper<Real>::setup_grids(params<Real> & p)
    {
        std::vector<hpx::future<void> > sum_futures;
        grids[src].resize(p.num_vars);
        grids[dst].resize(p.num_vars);
        for(std::size_t var = 0; var != num_vars; ++var)
        {
            grids[dst][var].resize(p.nx+2, p.ny+2, p.nz+2);
            grids[src][var].resize(p.nx+2, p.ny+2, p.nz+2);
            if(p.debug_grid)
            {
                for(std::size_t z = 0; z != nz + 2; ++z)
                {
                    for(std::size_t y = 0; y != ny + 2; ++y)
                    {
                        for(std::size_t x = 0; x != nx + 2; ++x)
                        {
                            grids[dst][var](x, y, z) = 0.0;
                            grids[src][var](x, y, z) = 0.0;
                        }
                    }
                }
                std::string filename = "initial_";
                filename += boost::lexical_cast<std::string>(rank);
                filename += "_";
                filename += boost::lexical_cast<std::string>(var);
                write_grid(grids[src][0], filename);
            }
            else
            {
                sum_futures.reserve(p.num_vars);
                Real sum = 0;
                for(std::size_t z = 0; z != nz + 2; ++z)
                {
                    for(std::size_t y = 0; y != ny + 2; ++y)
                    {
                        for(std::size_t x = 0; x != nx + 2; ++x)
                        {
                            Real value = random(gen);
                            grids[dst][var](x, y, z) = value;
                            grids[src][var](x, y, z) = value;
                            sum += value;
                        }
                    }
                }
                sum_futures.push_back(
                    global_sums[var].add(
                        hpx::util::bind(
                            set_global_sum_action()
                          , hpx::util::placeholders::_1 // The id to call the action on
                          , hpx::util::placeholders::_2 // The generation
                          , hpx::util::placeholders::_3 // Which rank calls this action
                          , hpx::util::placeholders::_4 // What value to sum
                          , var                   // Which partition to set
                        )
                      , stepper_ids
                      , rank
                      , sum
                    ).then(
                        [this, var](hpx::future<Real> value)
                        {
                            source_total[var] = value.get();
                        }
                    )
                );
            }
        }

        // Buff Init
        /////////////////////////////////////////////////////////////////////////
        num_sum_grid = 0;
        grids_to_sum.resize(p.num_vars, false);
        if(p.percent_sum == 100)
        {
            std::fill(grids_to_sum.begin(), grids_to_sum.end(), true);
            num_sum_grid = p.num_vars;
        }
        else if(p.percent_sum > 0)
        {
            double percent_sum = p.percent_sum / 100.0;
            for(std::size_t i = 0; i < p.num_vars; ++i)
            {
                double num = random(gen);
                if(num < percent_sum)
                {
                    grids_to_sum[i] = true;
                    ++num_sum_grid;
                }
            }
        }
        // Buff Init done ...
        /////////////////////////////////////////////////////////////////////////
        hpx::lcos::wait_all(sum_futures);
    }
*/

    template struct stepper<float>;
    template struct stepper<double>;
}
