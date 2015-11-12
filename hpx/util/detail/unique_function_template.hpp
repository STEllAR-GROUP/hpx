//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_UNIQUE_FUNCTION_TEMPLATE_HPP
#define HPX_UTIL_DETAIL_UNIQUE_FUNCTION_TEMPLATE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/detail/basic_function.hpp>
#include <hpx/util/detail/vtable/callable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <boost/mpl/identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Sig>
    struct unique_function_vtable_ptr
    {
        typename callable_vtable<Sig>::invoke_t invoke;
        vtable::get_type_t get_type;
        vtable::destruct_t destruct;
        vtable::delete_t delete_;
        bool empty;

        template <typename T>
        unique_function_vtable_ptr(boost::mpl::identity<T>) BOOST_NOEXCEPT
          : invoke(&callable_vtable<Sig>::template invoke<T>)
          , get_type(&vtable::template get_type<T>)
          , destruct(&vtable::template destruct<T>)
          , delete_(&vtable::template delete_<T>)
          , empty(std::is_same<T, empty_function<Sig> >::value)
        {}

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void construct(void** v, Arg&& arg)
        {
            vtable::construct<T>(v, std::forward<Arg>(arg));
        }

        template <typename T, typename Arg>
        BOOST_FORCEINLINE static void reconstruct(void** v, Arg&& arg)
        {
            vtable::reconstruct<T>(v, std::forward<Arg>(arg));
        }
    };
}}}

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <
        typename Sig
      , typename IAr = serialization::input_archive
      , typename OAr = serialization::output_archive
    >
    class unique_function;

    template <typename R, typename ...Ts, typename IAr, typename OAr>
    class unique_function<R(Ts...), IAr, OAr>
      : public detail::basic_function<
            detail::unique_function_vtable_ptr<R(Ts...)>
          , R(Ts...), IAr, OAr
        >
    {
        typedef detail::unique_function_vtable_ptr<R(Ts...)> vtable_ptr;
        typedef detail::basic_function<vtable_ptr, R(Ts...), IAr, OAr> base_type;

        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function);

    public:
        typedef typename base_type::result_type result_type;

        unique_function() BOOST_NOEXCEPT
          : base_type()
        {}

        unique_function(unique_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function(F&& f)
          : base_type()
        {
            assign(std::forward<F>(f));
        }

        unique_function& operator=(unique_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function& operator=(F&& f)
        {
            assign(std::forward<F>(f));
            return *this;
        }

        using base_type::operator();
        using base_type::assign;
        using base_type::reset;
        using base_type::empty;
        using base_type::target_type;
        using base_type::target;
    };

    template <typename Sig, typename IAr, typename OAr>
    static bool is_empty_function(
        unique_function<Sig, IAr, OAr> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

    ///////////////////////////////////////////////////////////////////////////
#   ifdef HPX_HAVE_CXX11_ALIAS_TEMPLATES

    template <typename Sig>
    using unique_function_nonser = unique_function<Sig, void, void>;

#   else

    template <typename T>
    class unique_function_nonser;

    template <typename R, typename ...Ts>
    class unique_function_nonser<R(Ts...)>
      : public unique_function<R(Ts...), void, void>
    {
        typedef unique_function<R(Ts...), void, void> base_type;

        HPX_MOVABLE_BUT_NOT_COPYABLE(unique_function_nonser);

    public:
        unique_function_nonser() BOOST_NOEXCEPT
          : base_type()
        {}

        unique_function_nonser(unique_function_nonser&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function_nonser>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function_nonser(F&& f)
          : base_type(std::forward<F>(f))
        {}

        unique_function_nonser& operator=(unique_function_nonser&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, unique_function_nonser>::value
             && traits::is_callable<FD&(Ts...), R>::value
            >::type>
        unique_function_nonser& operator=(F&& f)
        {
            base_type::operator=(std::forward<F>(f));
            return *this;
        }
    };

    template <typename Sig>
    static bool is_empty_function(
        unique_function_nonser<Sig> const& f) BOOST_NOEXCEPT
    {
        return f.empty();
    }

#   endif /*HPX_HAVE_CXX11_ALIAS_TEMPLATES*/
}}

#endif
