//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2009-2011 Matt Anderson
//  Copyright (c)      2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_DYNAMIC_STENCIL_VALUE_IMPL)
#define HPX_COMPONENTS_AMR_DYNAMIC_STENCIL_VALUE_IMPL

#include <boost/bind.hpp>
#include <boost/assert.hpp>

#include <algorithm>

#include <hpx/util/unlock_lock.hpp>

#include "dynamic_stencil_value.hpp"
#include "../functional_component.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    struct eval_helper
    {
        template <typename Adaptor>
        static int
        call(naming::id_type const& gid, naming::id_type const& value_gid,
            int row, int column, Adaptor &in, double cycle_time, parameter const& par)
        {
            std::vector<naming::id_type> input_gids(in.size());
            for (std::size_t i = 0; i < in.size(); ++i)
                input_gids[i] = in[i]->get_future().get();

            return components::amr::stubs::functional_component::eval(
                gid, value_gid, input_gids, row, column,cycle_time, par);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Lock>
    inline naming::id_type
    alloc_helper(Lock& l, naming::id_type const& gid, int row,
        parameter const& par)
    {
        double time = -1.0;
        std::vector<naming::id_type> placeholder;
        util::unlock_the_lock<Lock> ul(l);
        return components::amr::stubs::functional_component::alloc_data(
            gid, -1, -1, row,placeholder,time,par);
    }

    inline void
    free_helper_sync(naming::id_type& gid)
    {
        components::stubs::memory_block::free_sync(gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    inline dynamic_stencil_value::dynamic_stencil_value()
      : is_called_(false), driver_thread_(0), sem_result_(0),
        functional_gid_(naming::invalid_id), row_(-1), column_(-1),
        instencilsize_(-1), outstencilsize_(-1), mtx_("dataflow_dynamic_stencil_value")
    {
        std::fill(&value_gids_[0], &value_gids_[2], naming::invalid_id);

        // the threads driving the computation are created in
        // set_functional_component only (see below)
    }

    inline dynamic_stencil_value::~dynamic_stencil_value()
    {
    }

    inline void dynamic_stencil_value::finalize()
    {
       if (naming::invalid_id != value_gids_[1])
            free_helper_sync(value_gids_[1]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // The call action is used for the first time step only. It sets the
    // initial value and waits for the whole evolution to finish, returning the
    // result
    inline naming::id_type dynamic_stencil_value::call(naming::id_type const& initial)
    {
        start();

        is_called_ = true;

        // this needs to have been initialized
        if (std::size_t(-1) == instencilsize_ || std::size_t(-1) == outstencilsize_) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "dynamic_stencil_value::call",
                "this instance has not been initialized yet");
            return naming::invalid_id;
        }

        // sem_in_ is pre-initialized to 1, so we need to reset it
        for (std::size_t i = 0; i < outstencilsize_; ++i)
            sem_in_[i]->wait();

        // set new current value
        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(value_gids_[1] == naming::invalid_id);   // shouldn't be initialized yet
            value_gids_[1] = initial;
        }

        // signal all output threads it's safe to read value
        for (std::size_t i = 0; i < outstencilsize_; ++i)
            sem_out_[i]->signal();

        // wait for final result
        sem_result_.wait();

        // return the final value computed to the caller
        naming::id_type result = naming::invalid_id;

        {
            mutex_type::scoped_lock l(mtx_);
            std::swap(value_gids_[1], result);
        }

        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    // The main thread function loops through all operations of the time steps
    // to be handled by this instance:
    // - get values from previous time step
    // - calculate current result
    // - set the result for the next time step
    inline threads::thread_state dynamic_stencil_value::main()
    {
        // ask functional component to create the local data value
        {
            mutex_type::scoped_lock l(mtx_);
            value_gids_[0] = alloc_helper(l, functional_gid_, row_, par_);
        }

        // we need to store our current value gid/is_called_ on the stack,
        // because after is_last is true this object might have been destructed
        // already
        naming::id_type value_gid_to_be_freed = value_gids_[0];
        bool is_called = is_called_;

        // this is the main loop of the computation, gathering the values
        // from the previous time step, computing the result of the current
        // time step and storing the computed value in the memory_block
        // referenced by value_gid_
        int timesteps_to_go = 1;
        while (timesteps_to_go > 0) {
            // start acquire operations on input ports
            for (std::size_t i = 0; i < instencilsize_; ++i)
                in_[i]->aquire_value();         // non-blocking!

            // at this point all gid's have to be initialized
            BOOST_ASSERT(naming::invalid_id != functional_gid_);
            BOOST_ASSERT(naming::invalid_id != value_gids_[0]);

            // Compute the next value, store it in value_gids_[0]
            // The eval action returns an integer allowing to finish
            // computation (>0: still to go, 0: last step, <0: overdone)
            timesteps_to_go = eval_helper::call(functional_gid_,
                value_gids_[0], row_, column_, in_,cycle_time_, par_);

            // Wait for all output threads to have read the current value.
            // On the first time step the semaphore is preset to allow
            // to immediately set the value.
            for (std::size_t i = 0; i < outstencilsize_; ++i)
                sem_in_[i]->wait();

            // set new current value, allocate space for next current value
            // if needed (this may happen for all time steps except the first
            // one, where the first gets it's initial value during the
            // call_action)
            {
                mutex_type::scoped_lock l(mtx_);

                if (naming::invalid_id == value_gids_[1])
                    value_gids_[1] = alloc_helper(l, functional_gid_, row_, par_);

                std::swap(value_gids_[0], value_gids_[1]);
                value_gid_to_be_freed = value_gids_[0];
            }

            // signal all output threads it's safe to read value
            for (std::size_t i = 0; i < outstencilsize_; ++i)
                sem_out_[i]->signal();
        }

        if (is_called)
            sem_result_.signal();         // final result has been set
        free_helper_sync(value_gid_to_be_freed);

        return threads::thread_state(threads::terminated);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The function get will be called by the out-ports whenever
    /// the current value has been requested.
    inline naming::id_type dynamic_stencil_value::get_value(int i)
    {
        sem_out_[i]->wait();     // wait for the current value to be valid

        naming::id_type result = naming::invalid_id;
        {
            mutex_type::scoped_lock l(mtx_);
            result = value_gids_[1];  // acquire the current value
        }

        sem_in_[i]->signal();         // signal to have read the value
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline std::vector<naming::id_type> dynamic_stencil_value::get_output_ports()
    {
        mutex_type::scoped_lock l(mtx_);
        std::vector<naming::id_type> gids;

        // this needs to have been initialized
        if (std::size_t(-1) == instencilsize_ || std::size_t(-1) == outstencilsize_) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "dynamic_stencil_value::get_output_ports",
                "this instance has not been initialized yet");
            return gids;
        }

        gids.resize(outstencilsize_);
        for (std::size_t i = 0; i < outstencilsize_; ++i)
            gids[i] = out_[i];

        return gids;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline util::unused_type
    dynamic_stencil_value::connect_input_ports(
        std::vector<naming::id_type> const& gids)
    {
        // this needs to have been initialized
        if (std::size_t(-1) == instencilsize_ || std::size_t(-1) == outstencilsize_) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "dynamic_stencil_value::connect_input_ports",
                "this instance has not been initialized yet");
            return util::unused;
        }

        if (gids.size() < instencilsize_) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "dynamic_stencil_value::connect_input_ports",
                "insufficient number of gid's supplied");
            return util::unused;
        }

        for (std::size_t i = 0; i < instencilsize_; ++i)
            in_[i]->connect(gids[i]);

        return util::unused;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline util::unused_type
    dynamic_stencil_value::set_functional_component(naming::id_type const& gid,
        int row, int column, int instencilsize, int outstencilsize,
        double cycle_time,parameter const& par)
    {
        // store gid of functional component
        functional_gid_ = gid;
        row_ = row;
        column_ = column;
        instencilsize_ = instencilsize;
        outstencilsize_ = outstencilsize;
        cycle_time_ = cycle_time;
        par_ = par;

        sem_in_.resize(outstencilsize);
        sem_out_.resize(outstencilsize);
        in_.resize(instencilsize);
        out_.resize(outstencilsize);

        // create adaptors
        for (std::size_t i = 0; i < instencilsize_; ++i)
        {
            in_[i].reset(new in_adaptor_type());
        }
        for (std::size_t i = 0; i < outstencilsize_; ++i)
        {
            sem_in_[i].reset(new lcos::local::counting_semaphore(1));
            sem_out_[i].reset(new lcos::local::counting_semaphore());
            out_[i] = naming::id_type(
                components::server::create_one<out_adaptor_type>(
                    boost::bind(&dynamic_stencil_value::get_value, this, i)),
                    naming::id_type::managed);
        }
        return util::unused;
    }

    ///////////////////////////////////////////////////////////////////////////
    inline util::unused_type
    dynamic_stencil_value::start()
    {
        // if all inputs have been bound already we need to start the driver
        // thread
        if (0 == driver_thread_) {
            bool inputs_bound = true;
            for (std::size_t i = 0; i < instencilsize_ && inputs_bound; ++i)
                inputs_bound = in_[i]->is_bound();

            if (inputs_bound) {
                // run the thread which collects the input, executes the provided
                // functional element and sets the value for the next time step
                driver_thread_ = applier::register_thread(
                    boost::bind(&dynamic_stencil_value::main, this),
                    "dynamic_stencil_value::main");
            }
        }
        return util::unused;
    }

}}}}

#endif
