//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef HPX_TRAITS_NEEDS_AUTOMATIC_REGISTRATION_HPP
#define HPX_TRAITS_NEEDS_AUTOMATIC_REGISTRATION_HPP

#include <boost/mpl/bool.hpp>

namespace hpx { namespace traits
{
    // This trait is used to decide whether a class (or specialization) is
    // required to automatically register to the action factory
    template <typename T, typename Enable = void>
    struct needs_automatic_registration
      : boost::mpl::true_
    {};
}}

#endif
