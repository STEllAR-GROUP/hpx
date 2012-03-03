//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONSTRUCTOR_ARGUMENT_MAR_10_2010_0201PM)
#define HPX_CONSTRUCTOR_ARGUMENT_MAR_10_2010_0201PM

#include <boost/variant.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/variant.hpp>
#include <hpx/runtime/naming/name.hpp>

namespace hpx { namespace components
{
    ///////////////////////////////////////////////////////////////////////////
    // currently, we support a limited set of possible constructor argument
    // types
    typedef boost::variant<
        std::size_t
      , naming::gid_type
      , naming::id_type
    > constructor_argument;
}}

#endif


