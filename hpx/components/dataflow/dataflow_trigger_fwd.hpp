//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_LCOS_DATAFLOW_TRIGGER_FWD_HPP
#define HPX_LCOS_DATAFLOW_TRIGGER_FWD_HPP

#include <hpx/components/dataflow/is_dataflow.hpp>

namespace hpx { namespace lcos 
{
    struct dataflow_trigger;
}}

namespace hpx { namespace traits
{
    template <typename Dummy>
    struct is_dataflow<hpx::lcos::dataflow_trigger, Dummy>
        : boost::mpl::true_
    {};
}}

#endif
