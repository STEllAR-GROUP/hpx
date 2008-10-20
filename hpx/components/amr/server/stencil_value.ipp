//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM)
#define HPX_COMPONENTS_AMR_STENCIL_VALUE_IMPL_OCT_17_2008_0848AM

#include <hpx/components/amr/server/stencil_value.hpp>
#include <hpx/components/amr/server/functional_component_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace amr { namespace server 
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, int N>
    struct eval_helper;

    template <typename T>
    struct eval_helper<T, 1>
    {
        template <typename Adaptor>
        static T 
        call(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid, Adaptor* in)
        {
            typedef functional_component_base<T, 1>::eval_action action_type;
            lcos::eager_future<action_type, T> f(appl, gid, in[0]->get_result(self));
            return f.get_result(self); 
        }
    };

    template <typename T>
    struct eval_helper<T, 3>
    {
        template <typename Adaptor>
        static T
        call(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid, Adaptor* in)
        {
            typedef functional_component_base<T, 3>::eval_action action_type;
            lcos::eager_future<action_type, T> f(appl, gid, 
                in[0]->get_result(self), in[1]->get_result(self), 
                in[2]->get_result(self));
            return f.get_result(self); 
        }
    };

    template <typename T>
    struct eval_helper<T, 5>
    {
        template <typename Adaptor>
        static T 
        call(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid, Adaptor* in)
        {
            typedef functional_component_base<T, 5>::eval_action action_type;
            lcos::eager_future<action_type, T> f(appl, gid, 
                in[0]->get_result(self), in[1]->get_result(self), 
                in[2]->get_result(self), in[3]->get_result(self), 
                in[4]->get_result(self));
            return f.get_result(self); 
        }
    };

    template <typename T, int N>
    struct is_last_timestep_helper
    {
        static bool 
        call(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& gid)
        {
            typedef 
                functional_component_base<T, N>::is_last_timestep_action 
            action_type;

            lcos::eager_future<action_type, bool> f(appl, gid);
            return f.get_result(self); 
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, int N>
    stencil_value<T, N>::stencil_value(applier::applier& appl)
      : sem_in_(N), sem_out_(0)
    {
        // create adaptors
        for (std::size_t i = 0; i < N; ++i)
        {
            in_[i].reset(new in_adaptor_type(appl));
            out_[i].reset(new out_adaptor_type(appl));
            out_[i]->get()->set_callback(
                boost::bind(&stencil_value::get_value, this, _1, _2));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, int N>
    threads::thread_state  
    stencil_value<T, N>::call(threads::thread_self& self, 
        applier::applier& appl, result_type* result, T const& initial)
    {
        T next_value = initial;
        do {
            // start acquire operations on input ports
            for (std::size_t i = 0; i < N; ++i)
                in_[i]->aquire_value(appl);

            // wait for all output threads to have read the current value
            sem_in_.wait(self, N);

            // write new current value
            value_ = next_value;

            // signal all output threads it's safe to read value
            sem_out_.signal(self, N);

            // compute the next value
            next_value = eval_helper<result_type, N>::call(self, appl, 
                functional_gid_, in_);

        } while (!is_last_timestep_helper<result_type, N>::call(self, appl, functional_gid_));

        if (0 != result)
            *result = value_;

        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The function get_result will be called by the out-ports whenever 
    /// the current value has been requested.
    template <typename T, int N>
    void
    stencil_value<T, N>::get_value(threads::thread_self& self, 
        result_type* result)
    {
        sem_out_.wait(self, 1);     // wait for the current value to be valid
        *result = value_;           // acquire the current value
        sem_in_.signal(self, 1);    // signal to have read the value
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, int N>
    threads::thread_state 
    stencil_value<T, N>::get_output_ports(threads::thread_self& self, 
        applier::applier& appl, std::vector<naming::id_type> *gids)
    {
        gids->clear();
        for (std::size_t i = 0; i < N; ++i)
            gids->push_back(out_[i]->get_gid(appl));
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, int N>
    threads::thread_state 
    stencil_value<T, N>::connect_input_ports(threads::thread_self& self, 
        applier::applier& appl, std::vector<naming::id_type> const& gids)
    {
        if (gids.size() < N) {
            HPX_THROW_EXCEPTION(bad_parameter, 
                "insufficient numnber of gid's supplied");
            return threads::terminated;
        }
        for (std::size_t i = 0; i < N; ++i)
            in_[i]->connect(gids[i]);
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, int N>
    threads::thread_state 
    stencil_value<T, N>::set_functional_component(threads::thread_self& self, 
        applier::applier& appl, naming::id_type const& gid)
    {
        functional_gid_ = gid;
        return threads::terminated;
    }

}}}}

#endif
