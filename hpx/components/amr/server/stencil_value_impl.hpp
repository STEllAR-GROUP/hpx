//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM

#include <hpx/components/amr/server/stencil_value.hpp>
#include <hpx/components/amr/server/functional_component.hpp>

#include <boost/bind.hpp>
#include <boost/assert.hpp>
#include <boost/assign/std/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    struct eval_helper;

    template <>
    struct eval_helper<1>
    {
        template <typename Adaptor>
        static bool
        call(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid, naming::id_type const& value_gid, 
            Adaptor* in)
        {
            using namespace boost::assign;
            typedef typename functional_component::eval_action action_type;

            std::vector<naming::id_type> input_gids;
            input_gids += in[0]->get(self);

            lcos::eager_future<action_type, bool> f(appl, gid, value_gid, input_gids);
            return f.get(self); 
        }
    };

    template <>
    struct eval_helper<3>
    {
        template <typename Adaptor>
        static bool
        call(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid, naming::id_type const& value_gid, 
            Adaptor* in)
        {
            using namespace boost::assign;
            typedef typename functional_component::eval_action action_type;

            std::vector<naming::id_type> input_gids;
            input_gids += in[0]->get(self), in[1]->get(self), in[2]->get(self);

            lcos::eager_future<action_type, bool> f(appl, gid, value_gid, input_gids);
            return f.get(self); 
        }
    };

    template <>
    struct eval_helper<5>
    {
        template <typename Adaptor>
        static bool
        call(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid, naming::id_type const& value_gid, 
            Adaptor* in)
        {
            using namespace boost::assign;
            typedef typename functional_component::eval_action action_type;

            std::vector<naming::id_type> input_gids;
            input_gids += 
                in[0]->get(self), in[1]->get(self), 
                in[2]->get(self), in[3]->get(self), 
                in[4]->get(self);

            lcos::eager_future<action_type, bool> f(appl, gid, value_gid, input_gids);
            return f.get(self); 
        }
    };

    inline naming::id_type 
    init_helper(threads::thread_self& self, applier::applier& appl, 
        naming::id_type const& gid)
    {
        typedef functional_component::init_action action_type;

        lcos::eager_future<action_type, naming::id_type> f(appl, gid);
        return f.get(self); 
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    stencil_value<N>::stencil_value(applier::applier& appl)
      : sem_in_(N), sem_out_(0), sem_result_(0), 
        value_gid_(naming::invalid_id), backup_value_gid_(naming::invalid_id),
        functional_gid_(naming::invalid_id)
    {
        // create adaptors
        for (std::size_t i = 0; i < N; ++i)
        {
            in_[i].reset(new in_adaptor_type(appl));
            out_[i].reset(new out_adaptor_type(appl));
            out_[i]->get()->set_callback(
                boost::bind(&stencil_value::get_value, this, _1, _2));
        }
        // the threads driving the computation are created in 
        // set_functional_component only (see below)
    }

    ///////////////////////////////////////////////////////////////////////////
    // The call action is used for the first time step only. It sets the 
    // initial value and waits for the whole evolution to finish, returning the
    // result
    template <int N>
    threads::thread_state  
    stencil_value<N>::call(threads::thread_self& self, applier::applier& appl, 
        naming::id_type* result, naming::id_type const& initial)
    {
        // sem_in_ is pre-initialized to N, so we need to reset it
        sem_in_.wait(self, N);

        // set new current value
        value_gid_ = initial;

        // signal all output threads it's safe to read value
        sem_out_.signal(self, N);

        // wait for final result 
        sem_result_.wait(self, 1);

        *result = value_gid_;
        value_gid_ = naming::invalid_id;

        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    // The main thread function loops through all operations of the time steps
    // to be handled by this instance:
    // - get values from previous time step
    // - calculate current result
    // - set the result for the next time step
    template <int N>
    threads::thread_state  
    stencil_value<N>::main(threads::thread_self& self, applier::applier& appl)
    {
        bool is_last = false;
        while (!is_last) {
            // start acquire operations on input ports
            for (std::size_t i = 0; i < N; ++i)
                in_[i]->aquire_value(appl);         // non-blocking!

            // the valid_gid_ gets initialized for the first time step only
            if (naming::invalid_id == value_gid_)
                value_gid_ = init_helper(self, appl, functional_gid_);

            // at this point all gid's have to be initialized
            BOOST_ASSERT(naming::invalid_id != functional_gid_);
            BOOST_ASSERT(naming::invalid_id != value_gid_);
            BOOST_ASSERT(naming::invalid_id != backup_value_gid_);

            // Compute the next value, store it in backup_value_gid_
            // The eval action returns true for the last time step.
            is_last = eval_helper<N>::call(self, appl, functional_gid_, 
                backup_value_gid_, in_);

            // Wait for all output threads to have read the current value.
            // On the first time step the semaphore is preset to allow 
            // to immediately set the value.
            sem_in_.wait(self, N);

            // set new current value
            std::swap(value_gid_, backup_value_gid_);

            // signal all output threads it's safe to read value
            sem_out_.signal(self, N);
        }

        sem_result_.signal(self, 1);        // final result has been set
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The function get will be called by the out-ports whenever 
    /// the current value has been requested.
    template <int N>
    void
    stencil_value<N>::get_value(threads::thread_self& self, 
        naming::id_type* result)
    {
        sem_out_.wait(self, 1);     // wait for the current value to be valid
        *result = value_gid_;       // acquire the current value
        sem_in_.signal(self, 1);    // signal to have read the value
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    threads::thread_state 
    stencil_value<N>::get_output_ports(threads::thread_self& self, 
        applier::applier& appl, std::vector<naming::id_type> *gids)
    {
        gids->clear();
        for (std::size_t i = 0; i < N; ++i)
            gids->push_back(out_[i]->get_gid(appl));
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    threads::thread_state 
    stencil_value<N>::connect_input_ports(threads::thread_self& self, 
        applier::applier& appl, std::vector<naming::id_type> const& gids)
    {
        if (gids.size() < N) {
            HPX_THROW_EXCEPTION(bad_parameter, 
                "insufficient number of gid's supplied");
            return threads::terminated;
        }
        for (std::size_t i = 0; i < N; ++i)
            in_[i]->connect(gids[i]);
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <int N>
    threads::thread_state 
    stencil_value<N>::set_functional_component(threads::thread_self& self, 
        applier::applier& appl, naming::id_type const& gid)
    {
        // store gid of functional component
        functional_gid_ = gid;

        // ask functional component to create the local data value
        backup_value_gid_ = init_helper(self, appl, functional_gid_);

        // run the thread which collects the input, executes the provided
        // functional element and sets the value for the next time step
        applier::register_work(appl, 
            boost::bind(&stencil_value<N>::main, this, _1, boost::ref(appl)), 
            "stencil_value::main");

        return threads::terminated;
    }

}}}}

#endif
