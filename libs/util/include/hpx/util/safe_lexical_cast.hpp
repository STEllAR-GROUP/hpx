/*=============================================================================
    Copyright (c) 2014 Anton Bikineev

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/
#ifndef HPX_UTIL_SAFE_LEXICAL_CAST_HPP
#define HPX_UTIL_SAFE_LEXICAL_CAST_HPP

#include <hpx/config.hpp>

#include <boost/lexical_cast.hpp>

#include <string>
#include <type_traits>

namespace hpx { namespace util
{
    template <typename DestType, typename SrcType>
    DestType safe_lexical_cast(
        SrcType const& value, DestType const& dflt = DestType())
    {
        try
        {
            return boost::lexical_cast<DestType>(value);
        }
        catch (boost::bad_lexical_cast const&)
        {
            ;
        }
        return dflt;
    }

    template <typename DestType, typename Config>
    typename std::enable_if<
        std::is_integral<DestType>::value, DestType
    >::type get_entry_as(
        Config const& config, std::string const& key, DestType const& dflt)
    {
        return safe_lexical_cast(config.get_entry(key, dflt), dflt);
    }

    template <typename DestType, typename Config>
    DestType get_entry_as(
        Config const& config, std::string const& key, std::string const& dflt)
    {
        return safe_lexical_cast(config.get_entry(key, dflt),
            safe_lexical_cast<DestType>(dflt));
    }

}}

#endif /*HPX_UTIL_SAFE_LEXICAL_CAST_HPP*/
