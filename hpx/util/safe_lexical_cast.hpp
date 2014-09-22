/*=============================================================================
    Copyright (c) 2014 Anton Bikineev

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/
#if !defined(HPX_UTIL_SEP_21_2014_0840PM)
#define HPX_UTIL_SEP_21_2014_0840PM

#include <boost/lexical_cast.hpp>

namespace hpx { namespace util
{
    template <class DestType, class SrcType>
    DestType safe_lexical_cast(const SrcType& value, const DestType& dflt = DestType{})
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

}}

#endif //HPX_UTIL_SEP_21_2014_0840PM 
