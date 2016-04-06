//  Copyright (c) 2011 Matthew Anderson
//  Copyright (c) 2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include "allgather_and_gate.hpp"

#include <boost/bind.hpp>
#include <boost/ref.hpp>
#include <boost/format.hpp>
#include <boost/thread/locks.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace ag { namespace server
{
    void allgather_and_gate::init(
        std::vector<hpx::naming::id_type> const& components, std::size_t rank)
    {
        boost::lock_guard<mutex_type> l(mtx_);

        rank_ = rank;
        components_ = components;

        // prepare data array
        n_.clear();
        n_.resize(components.size());
    }

    void allgather_and_gate::compute(int num_loops)
    {
        for (int i = 0; i < num_loops; ++i)
        {
            // do some stuff
            double value = rank_ * 3.14159 * (i+1);

            // now hit the barrier
            allgather(value);
        }
    }

    void allgather_and_gate::allgather(double value)
    {
        // synchronize with all operations to finish
        std::size_t generation = 0;
        hpx::future<void> f = gate_.get_future(components_.size(), &generation);

        // Send our value to all participants of this allgather operation. We
        // assume components_, rank_ and value to be constant, thus no locking
        // is required.
        set_data_action set_data_;

        std::vector<hpx::id_type>::const_iterator end = components_.end();
        for (std::vector<hpx::id_type>::const_iterator it = components_.begin();
             it != end; ++it)
        {
            hpx::apply(set_data_, *it, rank_, generation, value);
        }

        // possibly do other stuff while the allgather is going on...

        f.get();
    }

    void allgather_and_gate::set_data(std::size_t which,
        std::size_t generation, double data)
    {
        gate_.synchronize(generation, "allgather_and_gate::set_data");

        {
            boost::lock_guard<mutex_type> l(mtx_);

            if (which >= n_.size())
            {
                // index out of bounds...
                HPX_THROW_EXCEPTION(hpx::bad_parameter,
                    "allgather_and_gate::set_data",
                    "index is out of range for this allgather operation");
                return;
            }
            n_[which] = data;         // set the received data
        }

        gate_.set(which);         // trigger corresponding and-gate input
    }

    void allgather_and_gate::print()
    {
        boost::lock_guard<mutex_type> l(mtx_);

        std::cout << " location: " << rank_ << " n size : " << n_.size() << std::endl;
        for (std::size_t i = 0; i < n_.size(); ++i)
        {
            std::cout << "     n value: " << n_[i] << std::endl;
        }
    }
}}

