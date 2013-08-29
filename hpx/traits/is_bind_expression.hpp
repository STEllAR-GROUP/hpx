//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_BIND_EXPRESSION_AUG_28_2013_0603PM)
#define HPX_TRAITS_IS_BIND_EXPRESSION_AUG_28_2013_0603PM

#include <boost/mpl/bool.hpp>

namespace hpx { namespace traits
{
    template <typename T>
    struct is_bind_expression
      : boost::mpl::false_
    {};
}}

#endif
