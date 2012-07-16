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
#include <boost/lexical_cast.hpp>

#include <iostream>
#include <fstream>

///////////////////////////////////////////////////////////////////////////////
namespace ag { namespace server
{
    void allgather_and_gate::compute(
        std::vector<hpx::naming::id_type> const& components, std::size_t rank,
        int num_loops)
    {
        rank_ = rank;
        components_ = components;

        for (int i = 0; i < num_loops; ++i)
        {
            // do some stuff
            double value = rank_ * 3.14159 * (i+1);

            // prepare data array
            n_.clear();
            n_.resize(components.size());

            // now hit the barrier
            allgather(value);
        }
    }

    void allgather_and_gate::allgather(double value)
    {
        mutex_type::scoped_lock l(mtx_);

        // create a new and-gate object
        gate_ = hpx::lcos::local::and_gate(components_.size());

        // In the future, this loop will be replaced with:
        //
        //    hpx::apply(set_data_, components_, value);
        //
        set_data_action set_data_;
        for (std::size_t i = 0; i < components_.size(); ++i)
        {
            hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            hpx::apply(set_data_, components_[i], rank_, value);
        }

        // synchronize with all operations to finish
        hpx::future<void> f = gate_.get_future();

        // possibly do other stuff while the allgather is going on...

        {
            hpx::util::unlock_the_lock<mutex_type::scoped_lock> ul(l);
            f.get();
        }

        gate_.reset();              // reset and-gate
    }

    void allgather_and_gate::set_data(std::size_t which, double data)
    {
        mutex_type::scoped_lock l(mtx_);

        if (which >= n_.size())
        {
            // index out of bounds...
            return;
        }

        n_[which] = data;         // set the received data
        gate_.set(which);         // trigger corresponding and-gate input
    }

    void allgather_and_gate::print()
    {
        mutex_type::scoped_lock l(mtx_);

        std::cout << " location: " << rank_ << " n size : " << n_.size() << std::endl;
        for (std::size_t i = 0; i < n_.size(); ++i)
        {
            std::cout << "     n value: " << n_[i] << std::endl;
        }
    }
}}

