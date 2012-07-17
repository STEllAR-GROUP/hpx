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
        // create a new and-gate object
        {
            mutex_type::scoped_lock l(mtx_);
            gate_ = hpx::lcos::local::and_gate(components_.size());
        }

        // Send our value to all participants of this allgather operation. We
        // assume components_, rank_ and value to be constant, thus no locking
        // is required.
        set_data_action set_data_;
        hpx::apply(set_data_, components_, rank_, value);

        // synchronize with all operations to finish
        hpx::future<void> f = gate_.get_future();

        // possibly do other stuff while the allgather is going on...

        f.get();

        // FIXME: reset operation might not be required
//         gate_.reset();              // reset and-gate
    }

    void allgather_and_gate::set_data(std::size_t which, double data)
    {
        mutex_type::scoped_lock l(mtx_);

        if (which >= n_.size())
        {
            // index out of bounds...
            HPX_THROW_EXCEPTION(hpx::bad_parameter,
                "allgather_and_gate::set_data",
                "index is out of range for this allgather operation");
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

