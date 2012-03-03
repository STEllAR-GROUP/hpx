//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_EXCEPTION_JAN_23_2009_0108PM)
#define HPX_UTIL_SERIALIZE_EXCEPTION_JAN_23_2009_0108PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <boost/config.hpp>
#include <boost/exception_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_free.hpp>

namespace hpx { namespace util
{
    enum exception_type
    {
        // unknown exception
        unknown_exception = 0,

        // standard exceptions
        std_runtime_error = 1,
        std_invalid_argument = 2,
        std_out_of_range = 3,
        std_logic_error = 4,
        std_bad_alloc = 5,
#ifndef BOOST_NO_TYPEID
        std_bad_cast = 6,
        std_bad_typeid = 7,
#endif
        std_bad_exception = 8,
        std_exception = 9,

        // boost exceptions
        boost_exception = 10,

        // boost::system::system_error
        boost_system_error = 11,

        // hpx::exception
        hpx_exception = 12
    };
}}  // namespace hpx::util

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace serialization
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void save(Archive& ar, boost::exception_ptr const& ep, unsigned int);

    ///////////////////////////////////////////////////////////////////////////
    template <typename Archive>
    void load(Archive& ar, boost::exception_ptr& e, unsigned int);
}}

BOOST_SERIALIZATION_SPLIT_FREE(boost::exception_ptr);

#endif
