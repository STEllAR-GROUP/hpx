//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file lcos/spli_all.hpp

#if !defined(HPX_LCOS_SPLIT_ALL_JUL_08_2016_0824AM)
#define HPX_LCOS_SPLIT_ALL_JUL_08_2016_0824AM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/detail/future_data.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/packaged_continuation.hpp>
#include <hpx/traits/acquire_future.hpp>
#include <hpx/traits/acquire_shared_state.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unused.hpp>

#include <boost/intrusive_ptr.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename ContResult>
        class split_continuation : public future_data<ContResult>
        {
            typedef future_data<ContResult> base_type;

        private:
            template <std::size_t I, typename T>
            void on_ready(
                typename traits::detail::shared_state_ptr_for<T>::type const&
                    state)
            {
                try {
                    this->base_type::set_value(hpx::util::get<I>(
                        traits::future_access<T>::create(state).get()));
                }
                catch (...) {
                    this->base_type::set_exception(boost::current_exception());
                }
            }

        public:
            template <std::size_t I, typename Future>
            void attach(Future& future)
            {
                typedef
                    typename traits::detail::shared_state_ptr_for<Future>::type
                    shared_state_ptr;

                // Bind an on_completed handler to this future which will wait
                // for the future and will transfer its result to the new
                // future.
                boost::intrusive_ptr<split_continuation> this_(this);
                shared_state_ptr const& state =
                    traits::detail::get_shared_state(future);

                state->execute_deferred();
                state->set_on_completed(util::deferred_call(
                    &split_continuation::on_ready<I, Future>, std::move(this_),
                    state));
            }
        };

        template <std::size_t I, typename ... Ts>
        inline typename hpx::traits::detail::shared_state_ptr<
            typename hpx::util::detail::at_index<I, Ts...>::type
        >::type
        extract_continuation(hpx::future<hpx::util::tuple<Ts...> >& future)
        {
            typedef typename hpx::util::detail::at_index<I, Ts...>::type
                result_type;
            typedef split_continuation<result_type> shared_state;

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p(new shared_state());

            static_cast<shared_state*>(p.get())->attach<I>(future);
            return p;
        }

        template <std::size_t I, typename ... Ts>
        HPX_FORCEINLINE
        future<typename hpx::util::detail::at_index<I, Ts...>::type>
        extract_nth_future(hpx::future<util::tuple<Ts...> >& future)
        {
            typedef typename hpx::util::detail::at_index<I, Ts...>::type
                result_type;

            return hpx::traits::future_access<hpx::future<result_type> >::
                create(extract_continuation<I>(future));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ... Ts, std::size_t ... Is>
        inline hpx::util::tuple<hpx::future<Ts>...>
        split_all_helper(hpx::future<hpx::util::tuple<Ts...> > && f,
            hpx::util::detail::pack_c<std::size_t, Is...>)
        {
            return hpx::util::make_tuple(extract_nth_future<Is>(f)...);
        }
    }

    template <typename ... Ts>
    inline hpx::util::tuple<future<Ts>...>
    split_all(hpx::future<hpx::util::tuple<Ts...> > && f)
    {
        return detail::split_all_helper(
                std::move(f),
                typename hpx::util::detail::make_index_pack<sizeof...(Ts)>::type()
            );
    }
}}

#endif
