//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM

#include <boost/bind.hpp>
#include <boost/assert.hpp>
#include <boost/assign/std/vector.hpp>

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
        static bool
        call(naming::id_type const& gid, naming::id_type const& value_gid, 
            Adaptor* in)
        {
            using namespace boost::assign;

            std::vector<naming::id_type> input_gids;
            input_gids += in[0]->get(), in[1]->get(), in[2]->get();

            return components::amr::stubs::functional_component::eval(
               gid, value_gid, input_gids);
        }
    };

    template <>
    struct eval_helper<5>
    {
        template <typename Adaptor>
        static bool
        call(naming::id_type const& gid, naming::id_type const& value_gid, 
            Adaptor* in)
        {
            using namespace boost::assign;

            std::vector<naming::id_type> input_gids;
            input_gids += 
                in[0]->get(), in[1]->get(), in[2]->get(), in[3]->get(), 
                in[4]->get();

            return components::amr::stubs::functional_component::eval(
                gid, value_gid, input_gids);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    inline naming::id_type 
    alloc_helper(naming::id_type const& gid)
    {
        return components::amr::stubs::functional_component::alloc_data(gid);
    }

    inline void free_helper(naming::id_type const& fgid, naming::id_type& gid)
    {
        components::amr::stubs::functional_component::free_data(fgid, gid);
        gid = naming::invalid_id;
    }

    inline void free_helper_sync(naming::id_type const& fgid, 
        naming::id_type& gid)
    {
        components::amr::stubs::functional_component::free_data_sync(fgid, gid);
        gid = naming::invalid_id;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    stencil_value<N>::stencil_value()
      : driver_thread_(0), sem_in_(N), sem_out_(0), sem_result_(0), 
        value_gid_(naming::invalid_id), backup_value_gid_(naming::invalid_id),
        functional_gid_(naming::invalid_id), is_called_(false)
    {
        // create adaptors
        for (std::size_t i = 0; i < N; ++i)
        {
            in_[i].reset(new in_adaptor_type());
            out_[i].reset(new out_adaptor_type());
            out_[i]->get()->set_callback(
                boost::bind(&stencil_value::get_value, this));
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
        if (naming::invalid_id != value_gid_) 
            free_helper_sync(functional_gid_, value_gid_);
    }

    ///////////////////////////////////////////////////////////////////////////
    // The call action is used for the first time step only. It sets the 
    // initial value and waits for the whole evolution to finish, returning the
    // result
    template <int N>
    naming::id_type stencil_value<N>::call(naming::id_type const& initial)
    {
        // remember that this instance is used as the first (and last) step in
        // the computation
        is_called_ = true;

        // sem_in_ is pre-initialized to N, so we need to reset it
        sem_in_.wait(N);

        // set new current value
        BOOST_ASSERT(value_gid_ == naming::invalid_id);   // shouldn't be initialized yet
        value_gid_ = initial;

        // signal all output threads it's safe to read value
        sem_out_.signal(N);

        // wait for final result 
        sem_result_.wait(1);

        // return the final value computed to the caller
        naming::id_type result = value_gid_;
        value_gid_ = naming::invalid_id;

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
        backup_value_gid_ = alloc_helper(functional_gid_);

        // this is the main loop of the computation, gathering the values
        // from the previous time step, computing the result of the current
        // time step and storing the computed value in the memory_block 
        // referenced by value_gid_
        bool is_last = false;
        while (!is_last) {
            // start acquire operations on input ports
            for (std::size_t i = 0; i < N; ++i)
                in_[i]->aquire_value();         // non-blocking!

            // at this point all gid's have to be initialized
            BOOST_ASSERT(naming::invalid_id != functional_gid_);
            BOOST_ASSERT(naming::invalid_id != backup_value_gid_);

            // Compute the next value, store it in backup_value_gid_
            // The eval action returns true for the last time step.
            is_last = eval_helper<N>::call(functional_gid_, 
                backup_value_gid_, in_);

            // if the computation finished in an instance which has been used
            // as the target for the initial call_action we can't exit right
            // away, because all other instances need to be executed one more 
            // time allowing them to free all their resources
            if (is_last && is_called_) {
                is_called_ = false;
                is_last = false;
            }

            // Wait for all output threads to have read the current value.
            // On the first time step the semaphore is preset to allow 
            // to immediately set the value.
            sem_in_.wait(N);

            // set new current value, allocate space for next current value
            // if needed (this may happen for all time steps except the first 
            // one, where the first gets it's initial value during the 
            // call_action)
            if (naming::invalid_id == value_gid_) 
                value_gid_ = alloc_helper(functional_gid_);
            std::swap(value_gid_, backup_value_gid_);

            // signal all output threads it's safe to read value
            sem_out_.signal(N);
        }

        sem_result_.signal(1);        // final result has been set

        free_helper_sync(functional_gid_, backup_value_gid_);
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The function get will be called by the out-ports whenever 
    /// the current value has been requested.
    template <int N>
    naming::id_type stencil_value<N>::get_value()
    {
        sem_out_.wait(1);     // wait for the current value to be valid
        naming::id_type result = value_gid_; // acquire the current value
        sem_in_.signal(1);    // signal to have read the value

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
    stencil_value<N>::set_functional_component(naming::id_type const& gid)
    {
        // store gid of functional component
        functional_gid_ = gid;

        // if all inputs have been bound already we need to start the driver 
        // thread
        bool inputs_bound = true;
        for (std::size_t i = 0; i < N && inputs_bound; ++i)
            inputs_bound = in_[i]->is_bound();

        if (inputs_bound && 0 == driver_thread_) {
            // run the thread which collects the input, executes the provided
            // functional element and sets the value for the next time step
            driver_thread_ = applier::register_thread(
                boost::bind(&stencil_value<N>::main, this), 
                "stencil_value::main");
        }
    }

}}}}

#endif
