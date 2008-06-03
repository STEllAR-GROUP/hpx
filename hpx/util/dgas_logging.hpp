//  Copyright (c) 2007-2008 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DGAS_LOGGING_APR_10_2008_1032AM)
#define HPX_UTIL_DGAS_LOGGING_APR_10_2008_1032AM

#include <boost/logging/format/named_write_fwd.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util 
{
    typedef boost::logging::named_logger<>::type logger_type;
    typedef boost::logging::level::holder filter_type;

    BOOST_DECLARE_LOG_FILTER(dgas_level, filter_type)
    BOOST_DECLARE_LOG(dgas_logger, logger_type)

    #define LDGAS_(lvl)                                                       \
        BOOST_LOG_USE_LOG_IF_LEVEL(util::dgas_logger(), util::dgas_level(), lvl)

    void init_dgas_logs();

    BOOST_DECLARE_LOG_FILTER(osh_level, filter_type)
    BOOST_DECLARE_LOG(osh_logger, logger_type)

    #define LOSH_(lvl)                                                        \
        BOOST_LOG_USE_LOG_IF_LEVEL(util::osh_logger(), util::osh_level(), lvl)

    void init_dgas_logs();

///////////////////////////////////////////////////////////////////////////////
}}

#endif


