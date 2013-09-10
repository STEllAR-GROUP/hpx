//  Copyright (c) 2011-2013 Thomas Heller
//  Copyright (c) 2011-2013 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_TUPLE_HELPER_HPP
#define HPX_UTIL_TUPLE_HELPER_HPP

#include <hpx/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/decay.hpp>

namespace hpx { namespace util
{
    namespace detail
    {
#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
        template <typename T, bool IsRvalueRef =
            std::is_rvalue_reference<T>::type::value>
#else
        template <typename T, bool IsRvalueRef = false>
#endif
        struct env_value_type
        {
            typedef typename hpx::util::detail::remove_reference<T>::type type;
        };

        template <typename T>
        struct env_value_type<T, false>
        {
            typedef T type;
        };

        template <typename T>
        struct env_value_type<T const, false>
        {
            typedef T const type;
        };

        template <typename T>
        struct env_value_type<T &, false>
        {
            typedef typename hpx::util::detail::remove_reference<T>::type & type;
        };

        template <typename T>
        struct env_value_type<T const &, false>
        {
            typedef typename hpx::util::detail::remove_reference<T>::type const & type;
        };

        struct ignore_type
        {
            template <typename T>
            ignore_type const& operator=(T const&) const
            {
                return *this;
            }
        };

// gcc 4.4.x is not able to cope with this, thus we disable the optimization
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 40500 || defined(BOOST_INTEL)
        ///////////////////////////////////////////////////////////////////////
        struct compute_sequence_is_bitwise_serializable
        {
            template <typename State, typename T>
            struct apply
              : boost::mpl::and_<
                    boost::serialization::is_bitwise_serializable<T>, State>
            {};
        };

        template <typename Seq>
        struct sequence_is_bitwise_serializable
          : boost::mpl::fold<
                Seq, boost::mpl::true_, compute_sequence_is_bitwise_serializable>
        {};
#else
        ///////////////////////////////////////////////////////////////////////
        template <typename Seq>
        struct sequence_is_bitwise_serializable
          : boost::mpl::false_
        {};
#endif
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_tuple
      : boost::mpl::false_
    {};

    template <typename T>
    struct is_tuple<T&>
      : is_tuple<T>
    {};

    template <typename T>
    struct is_tuple<T const&>
      : is_tuple<T>
    {};

    ///////////////////////////////////////////////////////////////////////////
    detail::ignore_type const ignore = {};
}}

#endif
