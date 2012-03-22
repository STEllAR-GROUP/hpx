//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <vector>
#include "sparse_matrix.hpp"

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/components/remote_object/new.hpp>
#include <hpx/components/dataflow/dataflow_object.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/async.hpp>
#include <hpx/include/iostreams.hpp>
#include <algorithm>

#include <hpx/components/dataflow/dataflow.hpp>
#include <hpx/components/dataflow/dataflow_trigger.hpp>
#include <hpx/components/dataflow/async_dataflow_wait.hpp>

#include <fstream>

#undef min

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;
using hpx::lcos::dataflow_trigger;
using hpx::lcos::dataflow_base;
using hpx::find_here;
using hpx::find_all_localities;
using hpx::find_here;
using hpx::lcos::wait;
using hpx::naming::id_type;
using hpx::naming::get_locality_from_id;

using hpx::components::new_;

using hpx::components::dataflow_object;

typedef std::pair<std::size_t, std::size_t> range_type;

struct lse_data
{
    lse_data(
        bright_future::crs_matrix<double> const & A
      , std::vector<double> const & x
      , std::vector<double> const & b
    )
        : A(A)
        , x(2, x)
        , b(b)
    {}

    bright_future::crs_matrix<double> A;
    std::vector<std::vector<double> > x;
    std::vector<double> b;
};

struct update_fun
{
    typedef void result_type;

    range_type range;
    std::size_t old;
    std::size_t n;

    update_fun() {}

    update_fun(update_fun const & other)
      : range(other.range)
      , old(other.old)
      , n(other.n)
    {}

    update_fun(BOOST_RV_REF(update_fun) other)
      : range(boost::move(other.range))
      , old(boost::move(other.old))
      , n(boost::move(other.n))
    {}

    update_fun(range_type const & r, std::size_t old, std::size_t n)
        : range(r)
        , old(old)
        , n(n)
    {}

    void operator()(lse_data & lse) const
    {
        jacobi_kernel_nonuniform(
            lse.A
          , lse.x
          , lse.b
          , range
          , old, n
        );
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & range;
        ar & old;
        ar & n;
    }

    private:
        BOOST_COPYABLE_AND_MOVABLE(update_fun)
};

struct return_
{
    typedef std::vector<double> result_type;

    std::size_t which;

    return_() {}
    return_(std::size_t which) : which(which) {}

    result_type operator()(lse_data & lse) const
    {
        return lse.x[which];
    }

    template <typename Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & which;
    }
};

void solve(
    bright_future::crs_matrix<double> const & A
  , std::vector<double> & x
  , std::vector<double> const & b_
  , std::size_t block_size
  , std::size_t max_iterations
)
{
    std::vector<range_type> block_ranges;
    for(std::size_t i = 0; i < x.size(); i += block_size)
    {
        block_ranges.push_back(
            range_type(i, std::min(x.size(), i + block_size))
        );
    }
    std::vector<std::vector<std::size_t> > dependencies(block_ranges.size());
    for(std::size_t b = 0; b < block_ranges.size(); ++b)
    {
        //std::cout << "checking for block " << b << "\n";
        for(std::size_t i = block_ranges[b].first; i < block_ranges[b].second; ++i)
        {
            std::size_t begin = A.row_begin(i);
            std::size_t end = A.row_end(i);

            for(std::size_t ii = begin; ii < end; ++ii)
            {
                std::size_t idx = A.indices[ii];
                for(std::size_t j = 0; j < block_ranges.size(); ++j)
                {
                    if(block_ranges[j].first <= idx && idx < block_ranges[j].second)
                    {
                        if(std::find(dependencies[b].begin(), dependencies[b].end(), j) == dependencies[b].end())
                        {
                            dependencies[b].push_back(j);
                        }
                        break;
                    }
                }
            }
        }
    }

    typedef dataflow_object<lse_data> object_type;

    object_type lse(new_<lse_data>(find_here(), A, x, b_).get());

    high_resolution_timer t;
    std::size_t old = 0;
    std::size_t n = 1;
    typedef dataflow_base<void> promise;
    typedef std::vector<promise> promise_grid_type;
    std::vector<promise_grid_type> deps(2, promise_grid_type(dependencies.size()));

    cout << "running " << max_iterations << " iterations\n" << flush;
    t.restart();
    for(std::size_t iter = 0; iter < max_iterations; ++iter)
    {
        for(std::size_t b = 0; b < block_ranges.size(); ++b)
        {
            if(iter > 0)
            {
                std::vector<promise > trigger(dependencies[b].size());
                for(std::size_t p = 0; p < dependencies[b].size(); ++p)
                {
                    trigger[p] = deps[old][dependencies[b][p]];
                }
                deps[n][b]
                    = lse.apply(
                        boost::move(update_fun(block_ranges[b], old, n))
                      , boost::move(dataflow_trigger(lse.gid_, boost::move(trigger)))
                    );
            }
            else
            {
                deps[n][b]
                    = lse.apply(
                        boost::move(update_fun(block_ranges[b], old, n))
                    );
            }
        }
        std::swap(old, n);
    }
    for(std::size_t i = 0; i < deps[0].size(); ++i)
    {
        deps[old][i].get_future().get();
    }

    double time_elapsed = t.elapsed();
    cout << x.size() << " "
         << (((x.size() * max_iterations)/1e6)/time_elapsed) << " MLUPS/s\n" << flush;

    x = lse.apply(return_(old)).get_future().get();
}
