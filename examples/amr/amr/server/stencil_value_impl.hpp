//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM

#include <boost/bind.hpp>
#include <boost/assert.hpp>
#include <boost/assign/std/vector.hpp>

#include <algorithm>

#include <hpx/util/unlock_lock.hpp>

#include "stencil_value.hpp"
#include "../functional_component.hpp"

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    struct eval_helper;

    template <>
    struct eval_helper<3>
    {
        template <typename Adaptor>
        static int
        call(naming::id_type const& gid, naming::id_type const& value_gid, 
            int row, int column, Adaptor* in)
        {
            using namespace boost::assign;

            std::vector<naming::id_type> input_gids;
            input_gids += in[0]->get(), in[1]->get(), in[2]->get();

            return components::amr::stubs::functional_component::eval(
               gid, value_gid, input_gids, row, column);
        }
    };

    template <>
    struct eval_helper<5>
    {
        template <typename Adaptor>
        static int
        call(naming::id_type const& gid, naming::id_type const& value_gid, 
            int row, int column, Adaptor* in)
        {
            using namespace boost::assign;

            std::vector<naming::id_type> input_gids;
            input_gids += 
                in[0]->get(), in[1]->get(), in[2]->get(), in[3]->get(), 
                in[4]->get();

            return components::amr::stubs::functional_component::eval(
                gid, value_gid, input_gids, row, column);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Lock>
    inline naming::id_type 
    alloc_helper(Lock& l, naming::id_type const& gid, int row)
    {
        util::unlock_the_lock<Lock> ul(l);
        return components::amr::stubs::functional_component::alloc_data(
            gid, -1, -1, row);
    }

    inline void 
    free_helper_sync(naming::id_type& gid)
    {
        components::stubs::memory_block::free_sync(gid);
        gid = naming::invalid_id;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    stencil_value<N>::stencil_value()
      : is_called_(false), driver_thread_(0), sem_result_(0), 
        functional_gid_(naming::invalid_id), row_(-1), column_(-1)
    {
        std::fill(&value_gids_[0], &value_gids_[2], naming::invalid_id);

        // create adaptors
        for (std::size_t i = 0; i < N; ++i)
        {
            in_[i].reset(new in_adaptor_type());
            out_[i].reset(new out_adaptor_type(
                boost::bind(&stencil_value<N>::get_value, this, i)));
        }

        // the threads driving the computation are created in 
        // set_functional_component only (see below)
    }

    template <int N>
    stencil_value<N>::~stencil_value()
    {
    }

    template <int N>
    void stencil_value<N>::finalize() 
    {
        if (naming::invalid_id != value_gids_[1]) 
            free_helper_sync(value_gids_[1]);
    }

    ///////////////////////////////////////////////////////////////////////////
    // The call action is used for the first time step only. It sets the 
    // initial value and waits for the whole evolution to finish, returning the
    // result
    template <int N>
    naming::id_type stencil_value<N>::call(naming::id_type const& initial)
    {
        is_called_ = true;

        // sem_in_ is pre-initialized to 1, so we need to reset it
        for (int i = 0; i < N; ++i)
            sem_in_[i].wait();

        // set new current value
        {
            mutex_type::scoped_lock l(mtx_);
            BOOST_ASSERT(value_gids_[1] == naming::invalid_id);   // shouldn't be initialized yet
            value_gids_[1] = initial;
        }

        // signal all output threads it's safe to read value
        for (int i = 0; i < N; ++i)
            sem_out_[i].signal();

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
    template <int N>
    threads::thread_state stencil_value<N>::main()
    {
        // ask functional component to create the local data value
        {
            mutex_type::scoped_lock l(mtx_);
            value_gids_[0] = alloc_helper(l, functional_gid_, row_);
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
            for (std::size_t i = 0; i < N; ++i)
                in_[i]->aquire_value();         // non-blocking!

            // at this point all gid's have to be initialized
            BOOST_ASSERT(naming::invalid_id != functional_gid_);
            BOOST_ASSERT(naming::invalid_id != value_gids_[0]);

            // Compute the next value, store it in value_gids_[0]
            // The eval action returns an integer allowing to finish 
            // computation (>0: still to go, 0: last step, <0: overdone)
            timesteps_to_go = eval_helper<N>::call(functional_gid_, 
                value_gids_[0], row_, column_, in_);

            // we're done if this is exactly the last time-step and we are not 
            // supposed to return the final value, no need to wait for further
            // input anymore
            if (timesteps_to_go < 0 && !is_called) {
                // exit immediately, 'this' might have been destructed already
                free_helper_sync(value_gid_to_be_freed);
                return threads::terminated;
            }

            // Wait for all output threads to have read the current value.
            // On the first time step the semaphore is preset to allow 
            // to immediately set the value.
            for (int i = 0; i < N; ++i)
                sem_in_[i].wait();

            // set new current value, allocate space for next current value
            // if needed (this may happen for all time steps except the first 
            // one, where the first gets it's initial value during the 
            // call_action)
            {
                mutex_type::scoped_lock l(mtx_);

                if (naming::invalid_id == value_gids_[1]) 
                    value_gids_[1] = alloc_helper(l, functional_gid_, row_);

                std::swap(value_gids_[0], value_gids_[1]);
                value_gid_to_be_freed = value_gids_[0];
            }

            // signal all output threads it's safe to read value
            for (int i = 0; i < N; ++i)
                sem_out_[i].signal();
        }

        sem_result_.signal();         // final result has been set
        free_helper_sync(value_gid_to_be_freed);

        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The function get will be called by the out-ports whenever 
    /// the current value has been requested.
    template <int N>
    naming::id_type stencil_value<N>::get_value(int i)
    {
        sem_out_[i].wait();     // wait for the current value to be valid

        naming::id_type result = naming::invalid_id;
        {
            mutex_type::scoped_lock l(mtx_);
            result = value_gids_[1]; // acquire the current value
        }

        sem_in_[i].signal();    // signal to have read the value
        return result;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    std::vector<naming::id_type> stencil_value<N>::get_output_ports()
    {
        std::vector<naming::id_type> gids;
        for (std::size_t i = 0; i < N; ++i)
            gids.push_back(out_[i]->get_gid());
        return gids;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    void stencil_value<N>::connect_input_ports(
        std::vector<naming::id_type> const& gids)
    {
        if (gids.size() < N) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "stencil_value<N>::connect_input_ports", 
                "insufficient number of gid's supplied");
            return;
        }
        for (std::size_t i = 0; i < N; ++i)
            in_[i]->connect(gids[i]);

        // if the functional component already has been set we need to start 
        // the driver thread
        if (functional_gid_ != naming::invalid_id && 0 == driver_thread_) {
            // run the thread which collects the input, executes the provided
            // functional element and sets the value for the next time step
            driver_thread_ = applier::register_thread(
                boost::bind(&stencil_value<N>::main, this), 
                "stencil_value::main");
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    void 
    stencil_value<N>::set_functional_component(naming::id_type const& gid,
        int row, int column)
    {
        // store gid of functional component
        functional_gid_ = gid;
        row_ = row;
        column_ = column;

        // if all inputs have been bound already we need to start the driver 
        // thread
        if (0 == driver_thread_) {
            bool inputs_bound = true;
            for (std::size_t i = 0; i < N && inputs_bound; ++i)
                inputs_bound = in_[i]->is_bound();

            if (inputs_bound) {
                // run the thread which collects the input, executes the provided
                // functional element and sets the value for the next time step
                driver_thread_ = applier::register_thread(
                    boost::bind(&stencil_value<N>::main, this), 
                    "stencil_value::main");
            }
        }
    }

}}}}

#endif
