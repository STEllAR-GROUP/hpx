//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_GHOST_PARTITION_HPP
#define HPX_EXAMPLES_MINI_GHOST_PARTITION_HPP

#include <examples/mini_ghost/global_sum.hpp>
#include <examples/mini_ghost/grid.hpp>
#include <examples/mini_ghost/params.hpp>
#include <examples/mini_ghost/recv_buffer.hpp>
#include <examples/mini_ghost/send_buffer.hpp>
#include <examples/mini_ghost/spikes.hpp>
#include <examples/mini_ghost/stencils.hpp>

#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/when_all.hpp>

namespace mini_ghost {
    template <
        typename Real
      , typename SumAction
      , typename SetSouthAction
      , typename SetNorthAction
      , typename SetWestAction
      , typename SetEastAction
      , typename SetFrontAction
      , typename SetBackAction
    >
    struct partition
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(partition);
    public:
        typedef grid<Real> grid_type;

        typedef
            hpx::util::serialize_buffer<Real>
            buffer_type;

        partition() {}

        template <typename Random>
        partition(
            std::size_t id
          , params<Real> & p
          , std::vector<hpx::id_type> & ids
          , std::size_t my_px
          , std::size_t my_py
          , std::size_t my_pz
          , Random const & random
          , hpx::future<void> & sum_future
        );

        partition(partition && rhs) = default;
        partition & operator=(partition && rhs) = default;

        void sum_allreduce(std::size_t which, std::size_t generation, Real value)
        {
            sum_allreduce_.set_data(which, generation, value);
        }

        bool sum_grid() const { return sum_grid_; }
        void insert_spike(spikes<Real> const & s, std::size_t spike)
        {
            Real source = s.insert(grids_[src_], id_, rank_, spike);
            s.insert(grids_[dst_], id_, rank_, spike);
            if(rank_ == 0)
            {
                source_total_ += source;
            }
        }

        void run(std::vector<hpx::id_type> & ids, std::size_t num_tsteps, hpx::lcos::local::spinlock & io_mutex)
        {
            hpx::future<void> sum_future = hpx::make_ready_future();
            hpx::future<double> flux_out_future = hpx::make_ready_future(0.0);
            std::vector<hpx::shared_future<void> > recv_futures(NUM_NEIGHBORS);

            std::size_t num_x_blocks = nx_ / nx_block_;
            if(nx_ % nx_block_) ++num_x_blocks;
            std::size_t num_y_blocks = ny_ / ny_block_;
            if(ny_ % ny_block_) ++num_y_blocks;
            std::size_t num_z_blocks = nz_ / nz_block_;
            if(nz_ % nz_block_) ++num_z_blocks;

            grid<hpx::shared_future<void> >
                calc_futures(num_x_blocks, num_y_blocks, num_z_blocks);

            for(std::size_t step = 1; step != num_tsteps+1; ++step)
            {
                sum_future = sum_future.then(
                    [
                        this
                      , &ids
                      , &io_mutex
                      , &flux_out_future
                      , &recv_futures
                      , &calc_futures
                      , step
                    ]
                    (hpx::future<void>)
                    {
                        std::vector<hpx::shared_future<void> > send_back_buffer_dependencies;
                        std::vector<hpx::shared_future<void> > send_front_buffer_dependencies;
                        std::vector<hpx::shared_future<void> > send_east_buffer_dependencies;
                        std::vector<hpx::shared_future<void> > send_west_buffer_dependencies;
                        std::vector<hpx::shared_future<void> > send_north_buffer_dependencies;
                        std::vector<hpx::shared_future<void> > send_south_buffer_dependencies;

                        // Receive our new buffers and copy them into the grid
                        if(step > 1)
                        {
                            recv_futures[BACK]
                                = recv_buffer_back_ (grids_[src_], step);
                            recv_futures[FRONT]
                                = recv_buffer_front_(grids_[src_], step);
                            recv_futures[EAST]
                                = recv_buffer_east_ (grids_[src_], step);
                            recv_futures[WEST]
                                = recv_buffer_west_ (grids_[src_], step);
                            recv_futures[NORTH]
                                = recv_buffer_north_(grids_[src_], step);
                            recv_futures[SOUTH]
                                = recv_buffer_south_(grids_[src_], step);
                        }

                        flux_out_future = flux_out_future.then(
                            hpx::util::bind(
                                [this](hpx::future<double> value, std::size_t src)
                                {
                                    return value.get() + flux_accumulate(src);
                                }
                              , hpx::util::placeholders::_1
                              , src_
                            )
                        );

                        for(std::size_t z = 1, nz_block = 0; z < nz_ + 1; z += nz_block_, ++nz_block)
                        {
                            range_type z_range(z, (std::min)(z + nz_block_, nz_ + 1));
                            for(std::size_t y = 1, ny_block = 0; y < ny_ + 1; y += ny_block_, ++ny_block)
                            {
                                range_type y_range(y, (std::min)(y + ny_block_, ny_ + 1));
                                for(std::size_t x = 1, nx_block = 0; x < nx_ + 1; x += nx_block_, ++nx_block)
                                {
                                    range_type x_range(x, (std::min)(x + nx_block_, nx_ + 1));
                                    HPX_ASSERT(nx_block < calc_futures.nx_);
                                    HPX_ASSERT(ny_block < calc_futures.ny_);
                                    HPX_ASSERT(nz_block < calc_futures.nz_);

                                    std::vector<hpx::shared_future<void> > dependencies;
                                    dependencies.reserve(NUM_NEIGHBORS + 1);
                                    // Setup calculation dependencies ...
                                    if(step > 1)
                                    {
                                        dependencies.push_back(calc_futures(nx_block, ny_block, nz_block));

                                        if(nx_block == 0)
                                        {
                                            if(recv_buffer_west_.valid_)
                                                dependencies.push_back(recv_futures[WEST]);
                                        }
                                        else
                                        {
                                            dependencies.push_back(calc_futures(nx_block-1, ny_block, nz_block));
                                        }

                                        if(nx_block == calc_futures.nx_-1)
                                        {
                                            if(recv_buffer_east_.valid_)
                                                dependencies.push_back(recv_futures[EAST]);
                                        }
                                        else
                                        {
                                            dependencies.push_back(calc_futures(nx_block+1, ny_block, nz_block));
                                        }

                                        if(ny_block == 0)
                                        {
                                            if(recv_buffer_south_.valid_)
                                                dependencies.push_back(recv_futures[SOUTH]);
                                        }
                                        else
                                        {
                                            dependencies.push_back(calc_futures(nx_block, ny_block-1, nz_block));
                                        }

                                        if(ny_block == calc_futures.ny_-1)
                                        {
                                            if(recv_buffer_north_.valid_)
                                                dependencies.push_back(recv_futures[NORTH]);
                                        }
                                        else
                                        {
                                            dependencies.push_back(calc_futures(nx_block, ny_block-1, nz_block));
                                        }

                                        if(nz_block == 0)
                                        {
                                            if(recv_buffer_front_.valid_)
                                                dependencies.push_back(recv_futures[FRONT]);
                                        }
                                        else
                                        {
                                            dependencies.push_back(calc_futures(nx_block, ny_block, nz_block-1));
                                        }

                                        if(nz_block == calc_futures.nz_-1)
                                        {
                                            if(recv_buffer_back_.valid_)
                                                dependencies.push_back(recv_futures[BACK]);
                                        }
                                        else
                                        {
                                            dependencies.push_back(calc_futures(nx_block, ny_block, nz_block+1));
                                        }
                                    }

                                    switch (stencil_)
                                    {
                                        case STENCIL_NONE:
                                            calc_futures(nx_block, ny_block, nz_block)
                                                = hpx::when_all(dependencies).then(
                                                    hpx::util::bind(
                                                        &stencils<STENCIL_NONE>::call<Real>
                                                      , boost::ref(grids_[dst_]), boost::ref(grids_[src_])
                                                      , x_range, y_range, z_range)
                                                    );
                                            break;
                                        case STENCIL_2D5PT:
                                            calc_futures(nx_block, ny_block, nz_block)
                                                = hpx::when_all(dependencies).then(
                                                    hpx::util::bind(
                                                    &stencils<STENCIL_2D5PT>::call<Real>
                                                  , boost::ref(grids_[dst_]), boost::ref(grids_[src_])
                                                  , x_range, y_range, z_range)
                                                );
                                            break;
                                        case STENCIL_2D9PT:
                                            calc_futures(nx_block, ny_block, nz_block)
                                                = hpx::when_all(dependencies).then(
                                                    hpx::util::bind(
                                                    &stencils<STENCIL_2D9PT>::call<Real>
                                                  , boost::ref(grids_[dst_]), boost::ref(grids_[src_])
                                                  , x_range, y_range, z_range)
                                                );
                                            break;
                                        case STENCIL_3D7PT:
                                            calc_futures(nx_block, ny_block, nz_block)
                                                = hpx::when_all(dependencies).then(
                                                    hpx::util::bind(
                                                    &stencils<STENCIL_3D7PT>::call<Real>
                                                  , boost::ref(grids_[dst_]), boost::ref(grids_[src_])
                                                  , x_range, y_range, z_range)
                                                );
                                            break;
                                        case STENCIL_3D27PT:
                                            calc_futures(nx_block, ny_block, nz_block)
                                                = hpx::when_all(dependencies).then(
                                                    hpx::util::bind(
                                                        &stencils<STENCIL_3D27PT>::call<Real>
                                                      , boost::ref(grids_[dst_]), boost::ref(grids_[src_])
                                                      , x_range, y_range, z_range)
                                                    );
                                            break;
                                        default:
                                            std::cerr << "Unknown stencil\n";
                                            hpx::terminate();
                                            break;
                                    }
                                    // Setup send buffer dependencies ...
                                    if(send_buffer_west_.dest_ && nx_block == 0)
                                    {
                                        send_west_buffer_dependencies
                                            .push_back(calc_futures(nx_block, ny_block, nz_block));
                                    }
                                    if(send_buffer_east_.dest_ && nx_block == calc_futures.nx_-1)
                                    {
                                        send_east_buffer_dependencies
                                            .push_back(calc_futures(nx_block, ny_block, nz_block));
                                    }
                                    if(send_buffer_south_.dest_ && ny_block == 0)
                                    {
                                        send_south_buffer_dependencies
                                            .push_back(calc_futures(nx_block, ny_block, nz_block));
                                    }
                                    if(send_buffer_north_.dest_ && ny_block == calc_futures.ny_-1)
                                    {
                                        send_north_buffer_dependencies
                                            .push_back(calc_futures(nx_block, ny_block, nz_block));
                                    }
                                    if(send_buffer_front_.dest_ && nz_block == 0)
                                    {
                                        send_front_buffer_dependencies
                                            .push_back(calc_futures(nx_block, ny_block, nz_block));
                                    }
                                    if(send_buffer_back_.dest_ && nz_block == calc_futures.nz_-1)
                                    {
                                        send_back_buffer_dependencies
                                            .push_back(calc_futures(nx_block, ny_block, nz_block));
                                    }
                                }
                            }
                        }

                        // Send boundaries ...
                        hpx::when_all(send_back_buffer_dependencies).then(
                            hpx::util::bind(
                                boost::ref(send_buffer_back_)
                              , boost::ref(grids_[dst_])
                              , step + 1
                              , id_
                            )
                        );
                        hpx::when_all(send_front_buffer_dependencies).then(
                            hpx::util::bind(
                                boost::ref(send_buffer_front_)
                              , boost::ref(grids_[dst_])
                              , step + 1
                              , id_
                            )
                        );
                        hpx::when_all(send_east_buffer_dependencies).then(
                            hpx::util::bind(
                                boost::ref(send_buffer_east_)
                              , boost::ref(grids_[dst_])
                              , step + 1
                              , id_
                            )
                        );
                        hpx::when_all(send_west_buffer_dependencies).then(
                            hpx::util::bind(
                                boost::ref(send_buffer_west_)
                              , boost::ref(grids_[dst_])
                              , step + 1
                              , id_
                            )
                        );
                        hpx::when_all(send_north_buffer_dependencies).then(
                            hpx::util::bind(
                                boost::ref(send_buffer_north_)
                              , boost::ref(grids_[dst_])
                              , step + 1
                              , id_
                            )
                        );
                        hpx::when_all(send_south_buffer_dependencies).then(
                            hpx::util::bind(
                                boost::ref(send_buffer_south_)
                              , boost::ref(grids_[dst_])
                              , step + 1
                              , id_
                            )
                        );

                        /*
                        std::string filename = "result_";
                        filename += boost::lexical_cast<std::string>(rank);
                        filename += "_";
                        filename += boost::lexical_cast<std::string>(step);
                        filename += "_";
                        filename += boost::lexical_cast<std::string>(var);
                        write_grid(grids_[dst_], filename);
                        */

                        std::swap(src_, dst_);
                        return sum_grid(ids, io_mutex, step, src_, flux_out_future, calc_futures.data_);
                    }
                );
            }
            sum_future.wait();
        }

        hpx::future<void> sum_grid(
            std::vector<hpx::id_type> & ids
          , hpx::lcos::local::spinlock & io_mutex
          , std::size_t step
          , std::size_t src
          , hpx::future<double> & flux_out_future
          , std::vector<hpx::shared_future<void> > & dependencies
        )
        {
            if(!sum_grid_)
            {
                return hpx::make_ready_future();
            }
            else
            {
                flux_out_ += flux_out_future.get();
                flux_out_future = hpx::make_ready_future(0.0);

                return
                    hpx::when_all(dependencies).then(
                        [this, src, step, &ids, &io_mutex](hpx::future<std::vector<hpx::shared_future<void>>>)
                        {
                            grid<Real> & g = grids_[src];
                            hpx::util::high_resolution_timer time_reduction;
                            // Sum grid ...
                            Real sum = 0;
                            for(std::size_t z = 1; z < g.nz_ - 1; ++z)
                            {
                                for(std::size_t y = 1; y < g.ny_ - 1; ++y)
                                {
                                    for(std::size_t x = 1; x < g.nx_ - 1; ++x)
                                    {
                                        sum += g(x, y, z);
                                    }
                                }
                            }
                            sum += flux_out_;

                            return
                                sum_allreduce_.add(
                                    hpx::util::bind(
                                        SumAction()
                                      , hpx::util::placeholders::_1 // The id to call the action on
                                      , hpx::util::placeholders::_2 // The generation
                                      , hpx::util::placeholders::_3 // Which rank calls this action
                                      , hpx::util::placeholders::_4 // What value to sum
                                      , id_                   // Which partition to set
                                    )
                                  , ids
                                  , rank_
                                  , sum
                                ).then(
                                    hpx::launch::sync,
                                    [this, step, &io_mutex](hpx::future<Real> value)
                                    {
                                        if(rank_ == 0)
                                        {
                                            // Report sum results for step-1
                                            Real error_iter
                                                = std::abs(source_total_ - value.get()) / source_total_;
                                            bool terminate = error_iter > error_tol_;

                                            if(terminate || (step % report_diffusion_ == 0))
                                            {
                                                hpx::lcos::local::spinlock::scoped_lock l(io_mutex);
                                                std::cout
                                                    << "Time step " << step << " "
                                                    << "for variable " << id_ << " "
                                                    << "the error is " << error_iter << "; "
                                                    << "error tolerance is " << error_tol_ << "."
                                                    << std::endl;
                                            }
                                            if(terminate) hpx::terminate();
                                        }
                                    }
                                );
                        }
                    );
            }
        }

        Real flux_accumulate(std::size_t src)
        {
            grid<Real> & g = grids_[src];
            Real divisor = 0.0;
            switch (stencil_)
            {
                case STENCIL_2D5PT:
                    divisor = 1.0f/5.0f;
                    break;
                case STENCIL_2D9PT:
                    divisor = 1.0f/9.0f;
                    break;
                case STENCIL_3D7PT:
                    divisor = 1.0f/7.0f;
                    break;
                case STENCIL_3D27PT:
                    divisor = 1.0f/27.0f;
                    break;
            }

            Real flux(0.0);
            if(my_px_ == 0)
            {
                for(std::size_t z = 1; z < g.nz_-1; ++z)
                {
                    for(std::size_t y = 1; y < g.ny_-1; ++y)
                    {
                        flux += g(1, y, z) * divisor;
                    }
                }
            }

            if(my_px_ == npx_ - 1)
            {
                for(std::size_t z = 1; z < g.nz_-1; ++z)
                {
                    for(std::size_t y = 1; y < g.ny_-1; ++y)
                    {
                        flux += g(g.nx_ - 1, y, z) * divisor;
                    }
                }
            }

            if(my_py_ == 0)
            {
                for(std::size_t z = 1; z < g.nz_-1; ++z)
                {
                    for(std::size_t x = 1; x < g.nx_-1; ++x)
                    {
                        flux += g(x, 1, z) * divisor;
                    }
                }
            }

            if(my_py_ == npy_ - 1)
            {
                for(std::size_t z = 1; z < g.nz_-1; ++z)
                {
                    for(std::size_t x = 1; x < g.nx_-1; ++x)
                    {
                        flux += g(x, g.ny_ - 1, z) * divisor;
                    }
                }
            }

            if(my_pz_ == 0)
            {
                for(std::size_t y = 1; y < g.ny_-1; ++y)
                {
                    for(std::size_t x = 1; x < g.nx_-1; ++x)
                    {
                        flux += g(x, y, 1) * divisor;
                    }
                }
            }

            if(my_pz_ == npz_ - 1)
            {
                for(std::size_t y = 1; y < g.ny_-1; ++y)
                {
                    for(std::size_t x = 1; x < g.nx_-1; ++x)
                    {
                        flux += g(x, y, g.nz_ - 1) * divisor;
                    }
                }
            }

            return flux;
        }

        send_buffer<buffer_type, NORTH, SetSouthAction> send_buffer_north_;
        recv_buffer<buffer_type, NORTH> recv_buffer_north_;

        send_buffer<buffer_type, SOUTH, SetNorthAction> send_buffer_south_;
        recv_buffer<buffer_type, SOUTH> recv_buffer_south_;

        send_buffer<buffer_type, EAST, SetWestAction> send_buffer_east_;
        recv_buffer<buffer_type, EAST> recv_buffer_east_;

        send_buffer<buffer_type, WEST, SetEastAction> send_buffer_west_;
        recv_buffer<buffer_type, WEST> recv_buffer_west_;

        send_buffer<buffer_type, FRONT, SetBackAction> send_buffer_front_;
        recv_buffer<buffer_type, FRONT> recv_buffer_front_;

        send_buffer<buffer_type, BACK, SetFrontAction> send_buffer_back_;
        recv_buffer<buffer_type, BACK> recv_buffer_back_;

    private:
        std::size_t id_;
        std::size_t stencil_;
        std::size_t src_;
        std::size_t dst_;
        std::array<grid_type, 2> grids_;

        global_sum<Real> sum_allreduce_;
        Real source_total_;
        Real flux_out_;

        std::size_t nx_;
        std::size_t ny_;
        std::size_t nz_;

        std::size_t nx_block_;
        std::size_t ny_block_;
        std::size_t nz_block_;

        std::size_t npx_;
        std::size_t npy_;
        std::size_t npz_;

        std::size_t my_px_;
        std::size_t my_py_;
        std::size_t my_pz_;

        std::size_t rank_;

        Real error_tol_;
        std::size_t report_diffusion_;
        bool sum_grid_;
    };

    template <
        typename Real
      , typename SumAction
      , typename SetSouthAction
      , typename SetNorthAction
      , typename SetWestAction
      , typename SetEastAction
      , typename SetFrontAction
      , typename SetBackAction
    >
    template <typename Random>
    partition<
        Real
      , SumAction
      , SetSouthAction
      , SetNorthAction
      , SetWestAction
      , SetEastAction
      , SetFrontAction
      , SetBackAction
    >::partition(
        std::size_t id
      , params<Real> & p
      , std::vector<hpx::id_type> & ids
      , std::size_t my_px
      , std::size_t my_py
      , std::size_t my_pz
      , Random const & random
      , hpx::future<void> & sum_future
    )
      : id_(id)
      , stencil_(p.stencil)
      , src_(0)
      , dst_(1)
      , nx_(p.nx)
      , ny_(p.ny)
      , nz_(p.nz)
      , nx_block_(p.nx_block)
      , ny_block_(p.ny_block)
      , nz_block_(p.nz_block)
      , npx_(p.npx)
      , npy_(p.npy)
      , npz_(p.npz)
      , my_px_(my_px)
      , my_py_(my_py)
      , my_pz_(my_pz)
      , rank_(p.rank)
      , error_tol_(p.error_tol)
      , report_diffusion_(p.report_diffusion)
      , sum_grid_(false)
    {
        grids_[dst_].resize(nx_+2, ny_+2, nz_+2, 0.0);
        grids_[src_].resize(nx_+2, ny_+2, nz_+2, 0.0);

        if(p.debug_grid)
        {
            std::string filename = "initial_";
            filename += boost::lexical_cast<std::string>(rank_);
            filename += "_";
            filename += boost::lexical_cast<std::string>(id_);
            write_grid(grids_[src_], filename);
            sum_future = hpx::make_ready_future();
        }
        else
        {
            Real sum = 0;
            for(std::size_t z = 0; z != nz_ + 2; ++z)
            {
                for(std::size_t y = 0; y != ny_ + 2; ++y)
                {
                    for(std::size_t x = 0; x != nx_ + 2; ++x)
                    {
                        Real value = random();
                        grids_[dst_](x, y, z) = value;
                        grids_[src_](x, y, z) = value;
                        sum += value;
                    }
                }
            }
            sum_future = sum_allreduce_.add(
                hpx::util::bind(
                    SumAction()
                  , hpx::util::placeholders::_1 // The id to call the action on
                  , hpx::util::placeholders::_2 // The generation
                  , hpx::util::placeholders::_3 // Which rank calls this action
                  , hpx::util::placeholders::_4 // What value to sum
                  , id_                         // Which partition to set
                )
              , ids
              , rank_
              , sum
            ).then(
                [this](hpx::future<Real> value)
                {
                    source_total_ = value.get();
                }
            );
        }

        if(p.percent_sum == 100)
        {
            sum_grid_ = true;
        }
        else if(p.percent_sum > 0)
        {
            double percent_sum = p.percent_sum / 100.0;
            double num = random();
            sum_grid_ = num < percent_sum;
        }

        // Setup neighbors
        if(my_py != 0)
        {
            send_buffer_south_.dest_ = ids[p.rank - p.npx];
            recv_buffer_south_.valid_ = true;
        }
        if(my_py != p.npy-1)
        {
            send_buffer_north_.dest_ = ids[p.rank + p.npx];
            recv_buffer_north_.valid_ = true;
        }
        if(my_px != 0)
        {
            send_buffer_west_.dest_ = ids[p.rank - 1];
            recv_buffer_west_.valid_ = true;
        }
        if(my_px != p.npx-1)
        {
            send_buffer_east_.dest_ = ids[p.rank + 1];
            recv_buffer_east_.valid_ = true;
        }
        if(my_pz != 0)
        {
            send_buffer_back_.dest_ = ids[p.rank - (p.npx*p.npy)];
            recv_buffer_back_.valid_ = true;
        }
        if(my_pz != p.npz-1)
        {
            send_buffer_front_.dest_ = ids[p.rank + (p.npx*p.npy)];
            recv_buffer_front_.valid_ = true;
        }
    }
}

#endif
