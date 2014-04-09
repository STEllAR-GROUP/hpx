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
            for(std::size_t step = 1; step != num_tsteps+1; ++step)
            {
                // Receive our new buffers and copy them into the grid
                if(step > 1)
                {
                    recv_buffer_back_ (grids_[src_], step);
                    recv_buffer_front_(grids_[src_], step);
                    recv_buffer_east_ (grids_[src_], step);
                    recv_buffer_west_ (grids_[src_], step);
                    recv_buffer_north_(grids_[src_], step);
                    recv_buffer_south_(grids_[src_], step);
                }

                // Wait on sum reduction ...
                sum_future.wait();

                flux_out_
                    += flux_accumulate(
                        grids_[src_], stencil_
                      , my_px_, my_py_, my_pz_
                      , npx_, npy_, npz_
                    );

                switch (stencil_)
                {
                    case STENCIL_NONE:
                        stencils<STENCIL_NONE>::call(
                            grids_[dst_], grids_[src_]);
                        break;
                    case STENCIL_2D5PT:
                        stencils<STENCIL_2D5PT>::call(
                            grids_[dst_], grids_[src_]);
                        break;
                    case STENCIL_2D9PT:
                        stencils<STENCIL_2D9PT>::call(
                            grids_[dst_], grids_[src_]);
                        break;
                    case STENCIL_3D7PT:
                        stencils<STENCIL_3D7PT>::call(
                            grids_[dst_], grids_[src_]);
                        break;
                    case STENCIL_3D27PT:
                        stencils<STENCIL_3D27PT>::call(
                            grids_[dst_], grids_[src_]);
                        break;
                    default:
                        std::cerr << "Unknown stencil\n";
                        hpx::terminate();
                        break;
                }

                // Send boundaries ...
                send_buffer_back_ (grids_[dst_], step+1, id_);
                send_buffer_front_(grids_[dst_], step+1, id_);
                send_buffer_east_ (grids_[dst_], step+1, id_);
                send_buffer_west_ (grids_[dst_], step+1, id_);
                send_buffer_north_(grids_[dst_], step+1, id_);
                send_buffer_south_(grids_[dst_], step+1, id_);

                /*
                std::string filename = "result_";
                filename += boost::lexical_cast<std::string>(rank);
                filename += "_";
                filename += boost::lexical_cast<std::string>(step);
                filename += "_";
                filename += boost::lexical_cast<std::string>(var);
                write_grid(grids_[dst_], filename);
                */

                if(sum_grid_)
                {
                    hpx::util::high_resolution_timer time_reduction;
                    // Sum grid ...
                    Real sum = 0;
                    for(std::size_t z = 1; z <= nz_; ++z)
                    {
                        for(std::size_t y = 1; y <= ny_; ++y)
                        {
                            for(std::size_t x = 1; x <= nx_; ++x)
                            {
                                sum += grids_[dst_](x, y, z);
                            }
                        }
                    }
                    sum += flux_out_;

                    sum_future =
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
                std::swap(src_, dst_);
            }
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
      , npx_(p.npx)
      , npy_(p.npy)
      , npz_(p.npz)
      , my_px_(my_px)
      , my_py_(my_py)
      , my_pz_(my_pz)
      , rank_(p.rank)
      , error_tol_(p.error_tol)
      , report_diffusion_(p.report_diffusion)
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
