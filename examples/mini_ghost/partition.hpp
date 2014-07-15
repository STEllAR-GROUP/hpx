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

#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/when_all.hpp>

namespace mini_ghost {
    template <
        typename Real
      , typename SumAction
      , typename SetSouthAction, typename SetNorthAction
      , typename SetWestAction, typename SetEastAction
      , typename SetFrontAction, typename SetBackAction
    >
    struct partition
    {
    private:
        HPX_MOVABLE_BUT_NOT_COPYABLE(partition);

    public:
        typedef grid<Real> grid_type;
        typedef hpx::util::serialize_buffer<Real> buffer_type;

        partition() {}

        template <typename Random>
        partition(
            std::size_t id
          , params<Real> & p
          , std::vector<hpx::id_type> & ids
          , std::vector<std::size_t> const & id_map
          , std::size_t my_px, std::size_t my_py, std::size_t my_pz
          , Random const & random
          , hpx::future<void> & sum_future
        );

        partition(partition &&other)
          : send_buffer_north_(std::move(other.send_buffer_north_))
          , recv_buffer_north_(std::move(other.recv_buffer_north_))
          , send_buffer_south_(std::move(other.send_buffer_south_))
          , recv_buffer_south_(std::move(other.recv_buffer_south_))
          , send_buffer_east_(std::move(other.send_buffer_east_))
          , recv_buffer_east_(std::move(other.recv_buffer_east_))
          , send_buffer_west_(std::move(other.send_buffer_west_))
          , recv_buffer_west_(std::move(other.recv_buffer_west_))
          , send_buffer_front_(std::move(other.send_buffer_front_))
          , recv_buffer_front_(std::move(other.recv_buffer_front_))
          , send_buffer_back_(std::move(other.send_buffer_back_))
          , recv_buffer_back_(std::move(other.recv_buffer_back_))
          , id_(other.id_)
          , stencil_(other.stencil_)
          , src_(other.src_)
          , dst_(other.dst_)
          , source_total_(other.source_total_)
          , flux_out_(other.flux_out_)
          , nx_(other.nx_)
          , ny_(other.ny_)
          , nz_(other.nz_)
          , nx_block_(other.nx_block_)
          , ny_block_(other.ny_block_)
          , nz_block_(other.nz_block_)
          , npx_(other.npx_)
          , npy_(other.npy_)
          , npz_(other.npz_)
          , my_px_(other.my_px_)
          , my_py_(other.my_py_)
          , my_pz_(other.my_pz_)
          , rank_(other.rank_)
          , error_tol_(other.error_tol_)
          , report_diffusion_(other.report_diffusion_)
          , sum_grid_(other.sum_grid_)
        {
            this->grids_[0] = std::move(other.grids_[0]);
            this->grids_[1] = std::move(other.grids_[1]);

            sum_allreduce_[0] = std::move(other.sum_allreduce_[0]);
            sum_allreduce_[1] = std::move(other.sum_allreduce_[1]);
        }
        partition& operator=(partition &&other)
        {
            recv_buffer_north_ = std::move(other.recv_buffer_north_);
            send_buffer_north_ = std::move(other.send_buffer_north_);
            recv_buffer_south_ = std::move(other.recv_buffer_south_);
            send_buffer_south_ = std::move(other.send_buffer_south_);
            recv_buffer_east_ = std::move(other.recv_buffer_east_);
            send_buffer_east_ = std::move(other.send_buffer_east_);
            recv_buffer_west_ = std::move(other.recv_buffer_west_);
            send_buffer_west_ = std::move(other.send_buffer_west_);
            recv_buffer_front_ = std::move(other.recv_buffer_front_);
            send_buffer_front_ = std::move(other.send_buffer_front_);
            recv_buffer_back_ = std::move(other.recv_buffer_back_);
            send_buffer_back_ = std::move(other.send_buffer_back_);
            this->id_      = other.id_;
            this->stencil_ = other.stencil_;
            this->src_     = other.src_;
            this->dst_     = other.dst_;
            this->grids_[0] = std::move(other.grids_[0]);
            this->grids_[1] = std::move(other.grids_[1]);

            sum_allreduce_[0] = std::move(other.sum_allreduce_[0]);
            sum_allreduce_[1] = std::move(other.sum_allreduce_[1]);
            this->source_total_  = other.source_total_;
            this->flux_out_      = other.flux_out_;

            this->nx_        = other.nx_;
            this->ny_        = other.ny_;
            this->nz_        = other.nz_;

            this->nx_block_  = other.nx_block_;
            this->ny_block_  = other.ny_block_;
            this->nz_block_  = other.nz_block_;

            this->npx_       = other.npx_;
            this->npy_       = other.npy_;
            this->npz_       = other.npz_;

            this->my_px_     = other.my_px_;
            this->my_py_     = other.my_py_;
            this->my_pz_     = other.my_pz_;

            this->rank_      = other.rank_;

            this->error_tol_         = other.error_tol_;
            this->report_diffusion_  = other.report_diffusion_;
            this->sum_grid_          = other.sum_grid_;
            return *this;
        }

        void sum_allreduce(std::size_t idx, std::size_t which, std::size_t generation, Real value)
        {
            sum_allreduce_[idx].set_data(which, generation, value);
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

        ///////////////////////////////////////////////////////////////////////
        void receive_boundaries(std::size_t step,
            std::vector<hpx::shared_future<void> >& recv_futures)
        {
            if(step > 1)
            {
                hpx::util::high_resolution_timer timer_recv;
                if(recv_buffer_back_.valid_)
                {
                    hpx::util::high_resolution_timer timer_recv_dir;
                    recv_futures[BACK] =
                        hpx::async(
                            hpx::util::bind(
                                boost::ref(recv_buffer_back_)
                                , boost::ref(grids_[src_])
                                , step
                            )
                        );
                    profiling::data().time_recv_z(timer_recv_dir.elapsed());
                }
                if(recv_buffer_front_.valid_)
                {
                    hpx::util::high_resolution_timer timer_recv_dir;
                    recv_futures[FRONT] =
                        hpx::async(
                            hpx::util::bind(
                                boost::ref(recv_buffer_front_)
                                , boost::ref(grids_[src_])
                                , step
                            )
                        );
                    profiling::data().time_recv_z(timer_recv_dir.elapsed());
                }
                if(recv_buffer_east_.valid_)
                {
                    hpx::util::high_resolution_timer timer_recv_dir;
                    recv_futures[EAST] =
                        hpx::async(
                            hpx::util::bind(
                                boost::ref(recv_buffer_east_)
                                , boost::ref(grids_[src_])
                                , step
                            )
                        );
                    profiling::data().time_recv_x(timer_recv_dir.elapsed());
                }
                if(recv_buffer_west_.valid_)
                {
                    hpx::util::high_resolution_timer timer_recv_dir;
                    recv_futures[WEST] =
                        hpx::async(
                            hpx::util::bind(
                                boost::ref(recv_buffer_west_)
                                , boost::ref(grids_[src_])
                                , step
                            )
                        );
                    profiling::data().time_recv_x(timer_recv_dir.elapsed());
                }
                if(recv_buffer_north_.valid_)
                {
                    hpx::util::high_resolution_timer timer_recv_dir;
                    recv_futures[NORTH] =
                        hpx::async(
                            hpx::util::bind(
                                boost::ref(recv_buffer_north_)
                                , boost::ref(grids_[src_])
                                , step
                            )
                        );
                    profiling::data().time_recv_y(timer_recv_dir.elapsed());
                }
                if(recv_buffer_south_.valid_)
                {
                    hpx::util::high_resolution_timer timer_recv_dir;
                    recv_futures[SOUTH] =
                        hpx::async(
                            hpx::util::bind(
                                boost::ref(recv_buffer_north_)
                                , boost::ref(grids_[src_])
                                , step
                            )
                        );
                    profiling::data().time_recv_y(timer_recv_dir.elapsed());
                }
                profiling::data().time_recv(timer_recv.elapsed());
            }
        }

        ///////////////////////////////////////////////////////////////////////
        hpx::shared_future<void> dependency_west(
            std::size_t nx_block, std::size_t ny_block, std::size_t nz_block,
            std::vector<hpx::shared_future<void> > const& recv_futures,
            grid<hpx::shared_future<void> > const& calc_futures)
        {
            if (nx_block == 0)
                return recv_futures[WEST];
            return calc_futures(nx_block-1, ny_block, nz_block);
        }

        hpx::shared_future<void> dependency_east(
            std::size_t nx_block, std::size_t ny_block, std::size_t nz_block,
            std::vector<hpx::shared_future<void> > const& recv_futures,
            grid<hpx::shared_future<void> > const& calc_futures)
        {
            if (nx_block == calc_futures.nx_-1)
                return recv_futures[EAST];
            return calc_futures(nx_block+1, ny_block, nz_block);
        }
        hpx::shared_future<void> dependency_south(
            std::size_t nx_block, std::size_t ny_block, std::size_t nz_block,
            std::vector<hpx::shared_future<void> > const& recv_futures,
            grid<hpx::shared_future<void> > const& calc_futures)
        {
            if (ny_block == 0)
                return recv_futures[SOUTH];
            return calc_futures(nx_block, ny_block-1, nz_block);
        }
        hpx::shared_future<void> dependency_north(
            std::size_t nx_block, std::size_t ny_block, std::size_t nz_block,
            std::vector<hpx::shared_future<void> > const& recv_futures,
            grid<hpx::shared_future<void> > const& calc_futures)
        {
            if (ny_block == calc_futures.ny_-1)
                return recv_futures[NORTH];
            return calc_futures(nx_block, ny_block+1, nz_block);
        }
        hpx::shared_future<void> dependency_front(
            std::size_t nx_block, std::size_t ny_block, std::size_t nz_block,
            std::vector<hpx::shared_future<void> > const& recv_futures,
            grid<hpx::shared_future<void> > const& calc_futures)
        {
            if (nz_block == 0)
                return recv_futures[FRONT];
            return calc_futures(nx_block, ny_block, nz_block-1);
        }
        hpx::shared_future<void> dependency_back(
            std::size_t nx_block, std::size_t ny_block, std::size_t nz_block,
            std::vector<hpx::shared_future<void> > const& recv_futures,
            grid<hpx::shared_future<void> > const& calc_futures)
        {
            if (nz_block == calc_futures.nz_-1)
                return recv_futures[BACK];
            return calc_futures(nx_block, ny_block, nz_block+1);
        }

        ///////////////////////////////////////////////////////////////////////
        void send_boundaries(std::size_t step,
            std::vector<std::vector<hpx::shared_future<void> > >& send_futures)
        {
            hpx::util::high_resolution_timer timer_send;
            if(!send_futures[BACK].empty())
            {
                hpx::util::high_resolution_timer timer_send_dir;
                hpx::when_all(send_futures[BACK]).then(
                    hpx::launch::async,
                    hpx::util::bind(
                        boost::ref(send_buffer_back_)
                        , boost::ref(grids_[dst_])
                        , step + 1
                        , id_
                    )
                );
                profiling::data().time_send_z(timer_send_dir.elapsed());
            }
            if(!send_futures[FRONT].empty())
            {
                hpx::util::high_resolution_timer timer_send_dir;
                hpx::when_all(send_futures[FRONT]).then(
                    hpx::launch::async,
                    hpx::util::bind(
                        boost::ref(send_buffer_front_)
                        , boost::ref(grids_[dst_])
                        , step + 1
                        , id_
                    )
                );
                profiling::data().time_send_z(timer_send_dir.elapsed());
            }
            if(!send_futures[EAST].empty())
            {
                hpx::util::high_resolution_timer timer_send_dir;
                hpx::when_all(send_futures[EAST]).then(
                    hpx::launch::async,
                    hpx::util::bind(
                        boost::ref(send_buffer_east_)
                        , boost::ref(grids_[dst_])
                        , step + 1
                        , id_
                    )
                );
                profiling::data().time_send_x(timer_send_dir.elapsed());
            }
            if(!send_futures[WEST].empty())
            {
                hpx::util::high_resolution_timer timer_send_dir;
                hpx::when_all(send_futures[WEST]).then(
                    hpx::launch::async,
                    hpx::util::bind(
                        boost::ref(send_buffer_west_)
                        , boost::ref(grids_[dst_])
                        , step + 1
                        , id_
                    )
                );
                profiling::data().time_send_x(timer_send_dir.elapsed());
            }
            if(!send_futures[NORTH].empty())
            {
                hpx::util::high_resolution_timer timer_send_dir;
                hpx::when_all(send_futures[NORTH]).then(
                    hpx::launch::async,
                    hpx::util::bind(
                        boost::ref(send_buffer_north_)
                        , boost::ref(grids_[dst_])
                        , step + 1
                        , id_
                    )
                );
                profiling::data().time_send_y(timer_send_dir.elapsed());
            }
            if(!send_futures[SOUTH].empty())
            {
                hpx::util::high_resolution_timer timer_send_dir;
                hpx::when_all(send_futures[SOUTH]).then(
                    hpx::launch::async,
                    hpx::util::bind(
                        boost::ref(send_buffer_south_)
                        , boost::ref(grids_[dst_])
                        , step + 1
                        , id_
                    )
                );
                profiling::data().time_send_y(timer_send_dir.elapsed());
            }
            profiling::data().time_send(timer_send.elapsed());
        }

        ///////////////////////////////////////////////////////////////////////
        // Main simulation loop
        //hpx::future<void>
        void run(std::vector<hpx::id_type> & ids, std::size_t num_tsteps,
            hpx::lcos::local::spinlock & io_mutex)
        {
            std::array<hpx::shared_future<void>, 2> sum_future;
            sum_future[src_] = hpx::make_ready_future();
            sum_future[dst_] = hpx::make_ready_future();

            std::array<hpx::shared_future<Real>, 2> flux_out_future;
            flux_out_future[src_] = hpx::make_ready_future(Real(0.0));
            flux_out_future[dst_] = hpx::make_ready_future(Real(0.0));

            std::vector<hpx::shared_future<void> > recv_futures(NUM_NEIGHBORS);
            for(hpx::shared_future<void> & f : recv_futures)
            {
                f = hpx::make_ready_future();
            }

            std::size_t num_x_blocks = nx_ / nx_block_;
            if(nx_ % nx_block_) ++num_x_blocks;
            std::size_t num_y_blocks = ny_ / ny_block_;
            if(ny_ % ny_block_) ++num_y_blocks;
            std::size_t num_z_blocks = nz_ / nz_block_;
            if(nz_ % nz_block_) ++num_z_blocks;

            std::array<grid<hpx::shared_future<void> >, 2> calc_futures;
            calc_futures[src_].resize(num_x_blocks, num_y_blocks, num_z_blocks);
            calc_futures[dst_].resize(num_x_blocks, num_y_blocks, num_z_blocks);

            std::vector<std::vector<hpx::shared_future<void> > > send_futures(NUM_NEIGHBORS);

            for(std::size_t step = 1; step != num_tsteps+1; ++step)
            {
                for(std::vector<hpx::shared_future<void> >& v: send_futures)
                {
                    v.clear();
                }

                // Receive boundaries asynchronously and copy them into the grid
                receive_boundaries(step, recv_futures);

                // calculate flux asynchronously once everything has been
                // received
                flux_out_future[dst_] =
                    hpx::when_all(recv_futures).then(
                        hpx::launch::async,
                        [this](hpx::future<void>)
                        {
                            return this->flux_accumulate(this->src_);
                        }
                    );

                // collect dependencies between partitions
                std::vector<hpx::shared_future<void> > dependencies;

                hpx::util::high_resolution_timer time_stencil;
                for(std::size_t z = 1, nz_block = 0; z < nz_ + 1; z += nz_block_, ++nz_block)
                {
                    range_type z_range(z, (std::min)(z + nz_block_, nz_ + 1));
                    for(std::size_t y = 1, ny_block = 0; y < ny_ + 1; y += ny_block_, ++ny_block)
                    {
                        range_type y_range(y, (std::min)(y + ny_block_, ny_ + 1));
                        for(std::size_t x = 1, nx_block = 0; x < nx_ + 1; x += nx_block_, ++nx_block)
                        {
                            range_type x_range(x, (std::min)(x + nx_block_, nx_ + 1));

                            HPX_ASSERT(nx_block < calc_futures[src_].nx_);
                            HPX_ASSERT(ny_block < calc_futures[src_].ny_);
                            HPX_ASSERT(nz_block < calc_futures[src_].nz_);

                            // Setup calculation dependencies ...
                            dependencies.clear();
                            dependencies.reserve(NUM_NEIGHBORS + 2);

                            if(step > 1)
                            {
                                dependencies.push_back(sum_future[src_]);
                                dependencies.push_back(calc_futures[src_](
                                    nx_block, ny_block, nz_block));

                                dependencies.push_back(dependency_west(
                                    nx_block, ny_block, nz_block,
                                    recv_futures, calc_futures[src_]));
                                dependencies.push_back(dependency_east(
                                    nx_block, ny_block, nz_block,
                                    recv_futures, calc_futures[src_]));
                                dependencies.push_back(dependency_south(
                                    nx_block, ny_block, nz_block,
                                    recv_futures, calc_futures[src_]));
                                dependencies.push_back(dependency_north(
                                    nx_block, ny_block, nz_block,
                                    recv_futures, calc_futures[src_]));
                                dependencies.push_back(dependency_front(
                                    nx_block, ny_block, nz_block,
                                    recv_futures, calc_futures[src_]));
                                dependencies.push_back(dependency_back(
                                    nx_block, ny_block, nz_block,
                                    recv_futures, calc_futures[src_]));
                            }

                            // launch stencil calculations asynchronously
                            hpx::shared_future<void> calc_future =
                                hpx::when_all(dependencies).then(
                                    hpx::launch::async,
                                    hpx::util::bind(
                                        get_stencil_call_op<Real>(stencil_)
                                      , boost::ref(grids_[dst_]), boost::ref(grids_[src_])
                                      , x_range, y_range, z_range)
                                    );
                            calc_futures[dst_](nx_block, ny_block, nz_block) =
                                calc_future;

                            // Setup send buffer dependencies ...
                            if(send_buffer_west_.dest_ && nx_block == 0)
                            {
                                send_futures[WEST].push_back(calc_future);
                            }
                            if(send_buffer_east_.dest_ && nx_block == calc_futures[dst_].nx_-1)
                            {
                                send_futures[EAST].push_back(calc_future);
                            }
                            if(send_buffer_south_.dest_ && ny_block == 0)
                            {
                                send_futures[SOUTH].push_back(calc_future);
                            }
                            if(send_buffer_north_.dest_ && ny_block == calc_futures[dst_].ny_-1)
                            {
                                send_futures[NORTH].push_back(calc_future);
                            }
                            if(send_buffer_front_.dest_ && nz_block == 0)
                            {
                                send_futures[FRONT].push_back(calc_future);
                            }
                            if(send_buffer_back_.dest_ && nz_block == calc_futures[dst_].nz_-1)
                            {
                                send_futures[BACK].push_back(calc_future);
                            }
                        }
                    }
                }

//                 profile_future = hpx::when_all(calc_futures[dst_].data_).then(
//                     hpx::launch::sync,
//                     [&time_stencil, this](hpx::future<std::vector<hpx::shared_future<void>>>)
//                     {
//                         profiling::data().time_stencil(time_stencil.elapsed());
//                         std::size_t num_pts = nx_ * ny_ * nz_;
//                         switch (stencil_)
//                         {
//                             case STENCIL_2D5PT:
//                                 profiling::data().num_adds(stencils<STENCIL_2D5PT>::num_adds*num_pts);
//                                 break;
//                             case STENCIL_2D9PT:
//                                 profiling::data().num_adds(stencils<STENCIL_2D9PT>::num_adds*num_pts);
//                                 break;
//                             case STENCIL_3D7PT:
//                                 profiling::data().num_adds(stencils<STENCIL_3D7PT>::num_adds*num_pts);
//                                 break;
//                             case STENCIL_3D27PT:
//                                 profiling::data().num_adds(stencils<STENCIL_3D27PT>::num_adds*num_pts);
//                                 break;
//                         }
//                         profiling::data().num_divides(num_pts);
//                     }
//                 );

                sum_future[dst_] = sum_grid(ids, io_mutex, step, dst_,
                    flux_out_future[dst_], calc_futures[dst_].data_);

                // Send boundaries ...
                send_boundaries(step, send_futures);

                std::swap(src_, dst_);
            }

            // FIXME: return future<void> from when_all
            hpx::wait_all(
                sum_future[dst_], sum_future[src_]
              , flux_out_future[dst_], flux_out_future[src_]
              , hpx::when_all(recv_futures)
              , hpx::when_all(calc_futures[dst_].data_)
              , hpx::when_all(calc_futures[src_].data_)
            );
        }

        ///////////////////////////////////////////////////////////////////////
        Real sum_all_grid(std::size_t dst) const
        {
            grid<Real> const& g = grids_[dst];
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
            return sum;
        }

        hpx::shared_future<void> sum_grid(
            std::vector<hpx::id_type> & ids
          , hpx::lcos::local::spinlock & io_mutex
          , std::size_t step
          , std::size_t dst
          , hpx::shared_future<Real> & flux_out_future
          , std::vector<hpx::shared_future<void> > & dependencies
        )
        {
            if(!sum_grid_)
                return hpx::shared_future<void>(flux_out_future);

            return
                hpx::lcos::local::dataflow(
                   hpx::launch::async,
                    [this, dst, step, &ids, &io_mutex](
                            std::vector<hpx::shared_future<void> > &&
                          , hpx::shared_future<Real> flux_out_future
                        ) -> hpx::future<void>
                    {
                        double time_now = hpx::util::high_resolution_timer::now();
                        hpx::util::high_resolution_timer timer_comp;

                        // Sum grid ...
                        Real sum(this->sum_all_grid(dst));
                        flux_out_ += flux_out_future.get();
                        sum += flux_out_;

                        profiling::data().time_sumgrid_comp(timer_comp.elapsed());

                        return
                            sum_allreduce_[dst].add(
                                SumAction(), ids, rank_, sum, id_, dst)
                            .then(
                                [this, time_now, step, &io_mutex](hpx::future<Real> value)
                                {
                                    hpx::util::high_resolution_timer timer(time_now);
                                    profiling::data().time_sumgrid_comm(timer.elapsed());
                                    profiling::data().time_sumgrid(timer.elapsed());
                                    profiling::data().num_sumgrid(1);
                                    if(rank_ != 0) return;

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
                            );
                    }
                  , dependencies
                  , flux_out_future
                ).share();
        }

        ///////////////////////////////////////////////////////////////////////
        Real get_divisor() const
        {
            switch (stencil_)
            {
                case STENCIL_NONE:
                    return Real(1.0);
                case STENCIL_2D5PT:
                    return Real(1.0)/Real(5.0);
                case STENCIL_2D9PT:
                    return Real(1.0)/Real(9.0);
                case STENCIL_3D7PT:
                    return Real(1.0)/Real(7.0);
                case STENCIL_3D27PT:
                    return Real(1.0)/Real(27.0);
                default:
                    HPX_ASSERT(false);
                    break;
            }
            return Real(1.0);
        }

        Real flux_accumulate(std::size_t src)
        {
            grid<Real> & g = grids_[src];
            Real const divisor = get_divisor();
            Real flux(0.0);

            if(my_px_ == 0)
            {
                for(std::size_t z = 1; z < g.nz_-1; ++z)
                {
                    for(std::size_t y = 1; y < g.ny_-1; ++y)
                    {
                        flux += g(0, y, z) * divisor;
                    }
                }
            }
            else if(my_px_ == npx_ - 1)
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
                        flux += g(x, 0, z) * divisor;
                    }
                }
            }
            else if(my_py_ == npy_ - 1)
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
                        flux += g(x, y, 0) * divisor;
                    }
                }
            }
            else if(my_pz_ == npz_ - 1)
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

        ///////////////////////////////////////////////////////////////////////
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

        std::array<global_sum<Real>, 2> sum_allreduce_;
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
        Real, SumAction
      , SetSouthAction, SetNorthAction
      , SetWestAction, SetEastAction
      , SetFrontAction, SetBackAction
    >::partition(
        std::size_t id
      , params<Real> & p
      , std::vector<hpx::id_type> & ids
      , std::vector<std::size_t> const & id_map
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
      , nx_(p.nx), ny_(p.ny), nz_(p.nz)
      , nx_block_(p.nx_block), ny_block_(p.ny_block), nz_block_(p.nz_block)
      , npx_(p.npx), npy_(p.npy), npz_(p.npz)
      , my_px_(my_px), my_py_(my_py), my_pz_(my_pz)
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
            for(std::size_t z = 1; z != nz_ + 1; ++z)
            {
                for(std::size_t y = 1; y != ny_ + 1; ++y)
                {
                    for(std::size_t x = 1; x != nx_ + 1; ++x)
                    {
                        Real value = random();
                        grids_[dst_](x, y, z) = value;
                        grids_[src_](x, y, z) = value;
                        sum += value;
                    }
                }
            }

            sum_future = sum_allreduce_[src_].add(
                    SumAction(), ids, rank_, sum, id_, src_)
               .then(
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
            sum_grid_ = ((id_ % 10) == 0);
        }

        // Setup neighbors
        if(my_py != 0)
        {
            send_buffer_south_.dest_ = ids[id_map[p.rank - p.npx]];
            recv_buffer_south_.valid_ = true;
        }
        if(my_py != p.npy-1)
        {
            send_buffer_north_.dest_ = ids[id_map[p.rank + p.npx]];
            recv_buffer_north_.valid_ = true;
        }
        if(my_px != 0)
        {
            send_buffer_west_.dest_ = ids[id_map[p.rank - 1]];
            recv_buffer_west_.valid_ = true;
        }
        if(my_px != p.npx-1)
        {
            send_buffer_east_.dest_ = ids[id_map[p.rank + 1]];
            recv_buffer_east_.valid_ = true;
        }
        if(my_pz != 0)
        {
            send_buffer_back_.dest_ = ids[id_map[p.rank - (p.npx*p.npy)]];
            recv_buffer_back_.valid_ = true;
        }
        if(my_pz != p.npz-1)
        {
            send_buffer_front_.dest_ = ids[id_map[p.rank + (p.npx*p.npy)]];
            recv_buffer_front_.valid_ = true;
        }
    }
}

#endif
