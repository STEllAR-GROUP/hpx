//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace server { namespace detail
{
    template <typename T, int N>
    struct call_helper;

    template <typename T>
    struct call_helper<T, 1>
    {
        template <typename Derived, typename Adaptor>
        static T eval(threads::thread_self& self, Derived& derived, Adaptor* in)
        {
            return derived.eval(in[0]->get_result(self));
        }
    };

    template <typename T>
    struct call_helper<T, 3>
    {
        template <typename Derived, typename Adaptor>
        static T eval(threads::thread_self& self, Derived& derived, Adaptor* in)
        {
            return derived.eval(in[0]->get_result(self), 
                in[1]->get_result(self), in[2]->get_result(self));
        }
    };

    template <typename T>
    struct call_helper<T, 5>
    {
        template <typename Derived, typename Adaptor>
        static T eval(threads::thread_self& self, Derived& derived, Adaptor* in)
        {
            return derived.eval(in[0]->get_result(self), 
                in[1]->get_result(self), in[2]->get_result(self), 
                in[3]->get_result(self), in[4]->get_result(self));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T, int N>
    threads::thread_state  
    stencil_value<Derived, T, N>::call(threads::thread_self& self, 
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
            next_value = call_helper<result_type, N>::eval(self, derived(), in_);

        } while (!derived().is_last_timestep());

        if (0 != result)
            *result = value_;

        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The function get_result will be called by the out-ports whenever 
    /// the current value has been requested.
    template <typename Derived, typename T, int N>
    void
    stencil_value<Derived, T, N>::get_value(threads::thread_self& self, 
        result_type* result)
    {
        sem_out_.wait(self, 1);     // wait for the current value to be valid
        *result = value_;           // acquire the current value
        sem_in_.signal(self, 1);    // signal to have read the value
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T, int N>
    threads::thread_state 
    stencil_value<Derived, T, N>::get_output_ports(threads::thread_self& self, 
        applier::applier& appl, std::vector<naming::id_type> *gids)
    {
        gids->clear();
        for (std::size_t i = 0; i < N; ++i)
            gids->push_back(out_[i]->get_gid(appl));
        return threads::terminated;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived, typename T, int N>
    threads::thread_state 
    stencil_value<Derived, T, N>::connect_input_ports(threads::thread_self& self, 
        applier::applier& appl, std::vector<naming::id_type> const& gids)
    {
        if (gids.size() < N) {
            HPX_THROW_EXCEPTION(bad_parameter, "insufficient gid's supplied");
            return threads::terminated;
        }
        for (std::size_t i = 0; i < N; ++i)
            in_[i]->connect(gids[i]);
        return threads::terminated;
    }

}}}}

