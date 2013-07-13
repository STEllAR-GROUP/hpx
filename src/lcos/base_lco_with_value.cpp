//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/util/ini.hpp>
#include <hpx/util/serialize_exception.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>


namespace hpx { namespace lcos
{
}}

///////////////////////////////////////////////////////////////////////////////
// FIXME: Do we still need these? Don't these auto-register?
//        For the time being: Yes we still need them
HPX_REGISTER_BASE_LCO_WITH_VALUE(hpx::naming::gid_type, gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE(std::vector<hpx::naming::gid_type>,
    vector_gid_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE(hpx::naming::id_type, id_type)
HPX_REGISTER_BASE_LCO_WITH_VALUE(std::vector<hpx::naming::id_type>,
    vector_id_type)
