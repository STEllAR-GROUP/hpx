//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_FUTURE_ACCESS_JUN_24_2015_0930AM)
#define HPX_TRAITS_FUTURE_ACCESS_JUN_24_2015_0930AM

#include <hpx/traits.hpp>
#include <hpx/traits/future_traits.hpp>

#include <boost/intrusive_ptr.hpp>
#include <boost/mpl/bool.hpp>

#include <type_traits>
#include <vector>

namespace hpx { namespace lcos
{
    template <typename R> class future;
    template <typename R> class shared_future;

    namespace detail
    {
        template <typename Result> struct future_data;
    }
}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename R>
        struct shared_state_ptr
        {
            typedef boost::intrusive_ptr<lcos::detail::future_data<R> > type;
        };

        template <typename Future>
        struct shared_state_ptr_for
          : shared_state_ptr<typename traits::future_traits<Future>::type>
        {};

        template <typename Future>
        struct shared_state_ptr_for<Future const>
          : shared_state_ptr_for<Future>
        {};

        template <typename Future>
        struct shared_state_ptr_for<Future&>
          : shared_state_ptr_for<Future>
        {};

        template <typename Future>
        struct shared_state_ptr_for<Future &&>
          : shared_state_ptr_for<Future>
        {};

        template <typename Future>
        struct shared_state_ptr_for<std::vector<Future> >
        {
            typedef typename traits::future_traits<Future>::type data_type;
            typedef std::vector<boost::intrusive_ptr<
                    lcos::detail::future_data<data_type>
                > > type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable>
    struct is_shared_state
      : boost::mpl::false_
    {};

    template <typename R>
    struct is_shared_state<boost::intrusive_ptr<lcos::detail::future_data<R> > >
      : boost::mpl::true_
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename R>
    struct future_access<lcos::future<R> >
    {
        template <typename SharedState>
        static lcos::future<R>
        create(boost::intrusive_ptr<SharedState> const& shared_state)
        {
            return lcos::future<R>(shared_state);
        }

        template <typename SharedState>
        static lcos::future<R>
        create(boost::intrusive_ptr<SharedState> && shared_state)
        {
            return lcos::future<R>(std::move(shared_state));
        }

        template <typename SharedState>
        static lcos::future<R>
        create(SharedState* shared_state)
        {
            return lcos::future<R>(
                boost::intrusive_ptr<SharedState>(shared_state));
        }

        HPX_FORCEINLINE static
        typename traits::detail::shared_state_ptr<R>::type const&
        get_shared_state(lcos::future<R> const& f)
        {
            return f.shared_state_;
        }
    };

    template <typename R>
    struct future_access<lcos::shared_future<R> >
    {
        template <typename SharedState>
        static lcos::shared_future<R>
        create(boost::intrusive_ptr<SharedState> const& shared_state)
        {
            return lcos::shared_future<R>(shared_state);
        }

        template <typename SharedState>
        static lcos::shared_future<R>
        create(boost::intrusive_ptr<SharedState> && shared_state)
        {
            return lcos::shared_future<R>(std::move(shared_state));
        }

        template <typename SharedState>
        static lcos::shared_future<R>
        create(SharedState* shared_state)
        {
            return lcos::shared_future<R>(
                boost::intrusive_ptr<SharedState>(shared_state));
        }

        HPX_FORCEINLINE static
        typename traits::detail::shared_state_ptr<R>::type const&
        get_shared_state(lcos::shared_future<R> const& f)
        {
            return f.shared_state_;
        }
    };
}}

#endif

