//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_IS_DATAFLOW_HPP
#define HPX_LCOS_IS_DATAFLOW_HPP

#include <boost/mpl/bool.hpp>

namespace hpx { namespace traits
{
    template <typename T, typename Dummy = void>
    struct is_dataflow
        : boost::mpl::false_
    {};
}}

#endif
