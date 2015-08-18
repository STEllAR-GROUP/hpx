/*=============================================================================
    Copyright (c) 2014 Anton Bikineev

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/
#if !defined(HPX_UTIL_SEP_21_2014_0840PM)
#define HPX_UTIL_SEP_21_2014_0840PM

#include <boost/lexical_cast.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    template <class DestType, class SrcType>
    DestType safe_lexical_cast(const SrcType& value, const DestType& dflt = DestType())
    {
        try
        {
            return boost::lexical_cast<DestType>(value);
        }
        catch (const boost::bad_lexical_cast& )
        {
            ;
        }
        return dflt;
    }

    template <class DestType, class Config>
    typename boost::enable_if<boost::is_integral<DestType>, DestType>::type
    get_entry_as(const Config& config, const std::string& key, const DestType& dflt)
    {
        return safe_lexical_cast(config.get_entry(key, dflt), dflt);
    }

    template <class DestType, class Config>
    DestType get_entry_as(const Config& config, const std::string& key,
        const std::string& dflt)
    {
        return safe_lexical_cast(config.get_entry(key, dflt),
            safe_lexical_cast<DestType>(dflt));
    }

}}

#endif //HPX_UTIL_SEP_21_2014_0840PM
