//  Copyright (c) 2016 Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_CALLABLE_WITH_HPP
#define HPX_TRAITS_IS_CALLABLE_WITH_HPP

#include <boost/type_traits/integral_constant.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        struct is_callable_with_impl
        {
            template<typename F, typename... Args>
            static decltype(std::declval<F>()(std::declval<Args>()...),
                boost::true_type())
            f(int);

            template<typename F, typename... Args>
            static boost::false_type
            f(...);
        };
    }
    
    ///////////////////////////////////////////////////////////////////////////
    template<typename F, typename... Args>
    using is_callable_with =
        decltype(detail::is_callable_with_impl::f<F, Args...>(0));

}}

#endif
