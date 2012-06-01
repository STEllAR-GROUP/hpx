
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/components/remote_object/distributed_new.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>

#include <hpx/include/iostreams.hpp>

#include "grid.hpp"
#include "row.hpp"

#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::util::high_resolution_timer;

using hpx::init;
using hpx::finalize;

using hpx::cout;
using hpx::flush;

namespace jacobi
{
    void stencil_row_update(row_range dst, row_range src_top, row_range src_center, row_range src_bottom)
    {
    }
}

HPX_PLAIN_ACTION(jacobi::stencil_row_update, jacobi_stencil_row_update)

namespace jacobi
{
    struct stencil_row
    {
        hpx::components::dataflow_object<stencil_row> top_;
        std::vector<hpx::components::dataflow_object<row> > center_;
        hpx::components::dataflow_object<stencil_row> bottom_;

        std::vector<hpx::lcos::dataflow_base<void> > dep;
        std::size_t nx;
        std::size_t src_;
        std::size_t dst_;

        stencil_row()
            : dep(2)
            , src_(0)
            , dst_(1)
        {}

        struct setup
        {
            typedef void result_type;
            hpx::components::dataflow_object<stencil_row> top_;
            std::vector<hpx::components::dataflow_object<row> > center_;
            hpx::components::dataflow_object<stencil_row> bottom_;
            std::size_t nx;

            setup() {}
            setup(
                std::vector<hpx::components::dataflow_object<row> > const & center
              , std::size_t n
            )
                : center_(center)
                , nx(n)
            {}
            
            setup(
                hpx::components::dataflow_object<stencil_row> const & top
              , std::vector<hpx::components::dataflow_object<row> > const & center
              , hpx::components::dataflow_object<stencil_row> const & bottom
              , std::size_t n
            )
                : top_(top)
                , center_(center)
                , bottom_(bottom)
                , nx(n)
            {}

            result_type operator()(
                stencil_row & s
            ) const
            {
                s.top_ = top_;
                s.center_ = center_;
                BOOST_ASSERT(s.center_.size() == 2);
                s.bottom_ = bottom_;
                s.nx = nx;

                BOOST_ASSERT(s.center_[0].gid_);
                BOOST_ASSERT(s.center_[1].gid_);
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & top_;
                ar & center_;
                ar & bottom_;
                ar & nx;
            }

        };

        struct get
        {
            typedef row_range result_type;

            std::size_t idx_;
            std::size_t begin_;
            std::size_t end_;

            get() {}

            get(std::size_t idx, std::size_t begin, std::size_t end)
                : idx_(idx)
                , begin_(begin)
                , end_(end)
            {}

            result_type operator()(stencil_row & s) const
            {
                /*
                BOOST_ASSERT(idx_ == 0 || idx_ == 1);
                BOOST_ASSERT(s.center_[idx_].gid_);
                */
                /*
                if(s.dep[s.src_].valid())
                {
                    return
                        //s.center_[idx_].apply(
                        s.center_[s.src_].apply(
                            ::jacobi::row::get(begin_, end_)
                          //, s.dep[idx_]
                          , s.dep[s.src_]
                        ).get_future().get();
                }
                else
                    */
                {
                    return
                        //s.center_[idx_].apply(
                        s.center_[s.src_].apply(
                            ::jacobi::row::get(begin_, end_)
                        ).get_future().get();
                }
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                ar & idx_;
                ar & begin_;
                ar & end_;
            }
        };

        struct update
        {
            typedef void result_type;

            update() {}

            result_type operator()(stencil_row & s) const
            {
                BOOST_ASSERT((s.src_ == 0 && s.dst_ == 1) || (s.src_ == 1 && s.dst_ == 0));
                BOOST_ASSERT(s.center_[s.src_].gid_);
                BOOST_ASSERT(s.center_[s.dst_].gid_);
                BOOST_ASSERT(s.top_.gid_);
                BOOST_ASSERT(s.bottom_.gid_);
                BOOST_ASSERT(s.dep.size() == 2);
                /*
                if(s.dep[s.src_].valid())
                {
                    s.dep[s.dst_] =
                        hpx::lcos::dataflow<jacobi_stencil_row_update>(
                            hpx::naming::get_locality_from_id(s.center_[s.dst_].gid_)
                          , s.center_[s.dst_].apply(row::get(0, s.nx))
                          , s.top_.apply(stencil_row::get(s.src_, 0, s.nx))
                          , s.center_[s.src_].apply(row::get(0, s.nx))
                          , s.bottom_.apply(stencil_row::get(s.src_, 0, s.nx))
                          , s.dep[s.src_]
                        );
                }
                else
                    */
                {
                    s.dep[s.dst_] =
                        hpx::lcos::dataflow<jacobi_stencil_row_update>(
                            hpx::naming::get_locality_from_id(s.center_[s.dst_].gid_)
                          , s.center_[s.dst_].apply(
                                ::jacobi::row::get(0, s.nx)
                            )
                          , s.top_.apply(
                                ::jacobi::stencil_row::get(s.src_, 0, s.nx)
                            )
                          , s.center_[s.src_].apply(
                                ::jacobi::row::get(0, s.nx)
                            )
                          , s.bottom_.apply(
                                ::jacobi::stencil_row::get(s.src_, 0, s.nx)
                            )
                        );
                }
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
            }
        };

        struct swap
        {
            typedef void result_type;

            swap() {}

            result_type operator()(stencil_row & s) const
            {
                std::swap(s.src_, s.dst_);
            }
            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
            }
        };

        struct wait
        {
            wait() {}

            typedef void result_type;
            result_type operator()(stencil_row & s) const
            {
                if(s.dep[s.src_].valid())
                    s.dep[s.src_].get_future().get();
                if(s.dep[s.dst_].valid())
                    s.dep[s.dst_].get_future().get();
            }

            template <typename Archive>
            void serialize(Archive & ar, unsigned)
            {
                //ar & idx_;
            }
        };
    };
}

int hpx_main(variables_map & vm)
{
    {
        std::size_t nx = vm["nx"].as<std::size_t>();
        std::size_t ny = vm["ny"].as<std::size_t>();
        std::size_t max_iterations = vm["max_iterations"].as<std::size_t>();

        std::vector<jacobi::grid> u(2, jacobi::grid(nx, ny, 1.0));

        std::vector<hpx::components::dataflow_object<jacobi::stencil_row> > stencil_rows(ny);

        {
            auto stencil_row_futures = hpx::components::distributed_new<jacobi::stencil_row>(ny);
            std::vector<hpx::components::dataflow_object<jacobi::row> > rows(2);

            stencil_rows[0] = stencil_row_futures[0].get();
            rows[0] = u[0].rows[0];
            rows[1] = u[1].rows[0];
            BOOST_ASSERT(rows[0].gid_);
            BOOST_ASSERT(rows[1].gid_);
            stencil_rows[0].apply(jacobi::stencil_row::setup(rows, nx)).get_future().get();

            stencil_rows[1] = stencil_row_futures[0].get();
            for(std::size_t idx = 1; idx < ny -1; ++idx)
            {
                stencil_rows[idx + 1] = stencil_row_futures[idx].get();
                BOOST_ASSERT(stencil_rows[idx].gid_);
                rows[0] = u[0].rows[idx];
                rows[1] = u[1].rows[idx];
                BOOST_ASSERT(rows[0].gid_);
                BOOST_ASSERT(rows[1].gid_);
                BOOST_ASSERT(stencil_rows[idx-1].gid_);
                BOOST_ASSERT(stencil_rows[idx+1].gid_);
                stencil_rows[idx].apply(jacobi::stencil_row::setup(stencil_rows[idx-1], rows, stencil_rows[idx+1], nx)).get_future().get();
            }

            //stencil_rows[ny-1] = stencil_row_futures[ny-1].get();
            rows[0] = u[0].rows[ny-1];
            rows[1] = u[1].rows[ny-1];
            BOOST_ASSERT(rows[0].gid_);
            BOOST_ASSERT(rows[1].gid_);
            stencil_rows[ny-1].apply(jacobi::stencil_row::setup(rows, nx)).get_future().get();
        }

        high_resolution_timer t;
        t.restart();
        std::vector<hpx::lcos::dataflow_base<void> > iter_deps(ny-2);
        for(std::size_t iter = 0; iter < max_iterations; ++iter)
        {
            for(std::size_t y = 1; y < ny - 1; ++y)
            {
                if(iter_deps[y-1].valid())
                {
                    iter_deps[y-1] = 
                        stencil_rows[y].apply(
                            jacobi::stencil_row::update()
                          , iter_deps[y-1]
                        );
                }
                else
                {
                    iter_deps[y-1] = 
                        stencil_rows[y].apply(
                            jacobi::stencil_row::update()
                        );
                }
                //iter_deps[y-1].get_future().get();
            }
            for(std::size_t y = 0; y < ny - 2; ++y)
            {
                std::vector<hpx::lcos::dataflow_base<void> > swap_trigger;

                swap_trigger.reserve(3);
                if(y > 0)
                    swap_trigger.push_back(iter_deps[y-1]);

                swap_trigger.push_back(iter_deps[y]);

                if(y  + 1 < ny - 2)
                    swap_trigger.push_back(iter_deps[y+1]);

                iter_deps[y-1] = 
                    stencil_rows[y].apply(
                        jacobi::stencil_row::swap()
                      , hpx::lcos::dataflow_trigger(stencil_rows[y].gid_, swap_trigger)
                    );
            }
        }
        // wait for everything to complete ...
        for(std::size_t y = 1; y < ny - 1; ++y)
        {
            stencil_rows[y].apply(jacobi::stencil_row::wait()).get_future().get();
            iter_deps[y-1].get_future().get();
        }
        
        double time_elapsed = t.elapsed();
        cout << nx << "x" << ny << " "
             << ((double((nx-2)*(ny-2) * max_iterations)/1e6)/time_elapsed) << " MLUP/S\n" << flush;

        finalize();
    }

    return 0;
}

int main(int argc, char **argv)
{
    options_description
        desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");
    
    desc_commandline.add_options()
        (
            "output"
          , value<std::string>()
          , "Output results to file"
        )
        (
            "nx"
          , value<std::size_t>()->default_value(10)
          , "Number of elements in x direction (columns)"
        )
        (
            "ny"
          , value<std::size_t>()->default_value(10)
          , "Number of elements in y direction (rows)"
        )
        (
            "max_iterations"
          , value<std::size_t>()->default_value(10)
          , "Maximum number of iterations"
        )
        (
            "line_block"
          , value<std::size_t>()->default_value(10)
          , "Number of line elements to block the iteration"
        )
        ;

    return init(desc_commandline, argc, argv);
}
