//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_LCOS_DETAIL_FUTURE_TRAITS_DEC_05_2017_0214PM)
#define HPX_LCOS_DETAIL_FUTURE_TRAITS_DEC_05_2017_0214PM

#include <hpx/config.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/traits/future_traits.hpp>

#include <iterator>

namespace hpx { namespace lcos { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename Enable = void>
    struct future_iterator_traits
    {};

    template <typename Iterator>
    struct future_iterator_traits<Iterator,
        typename hpx::util::always_void<
            typename std::iterator_traits<Iterator>::value_type
        >::type>
    {
        typedef
            typename std::iterator_traits<Iterator>::value_type
            type;

        typedef hpx::traits::future_traits<type> traits_type;
    };
}}}

#endif
