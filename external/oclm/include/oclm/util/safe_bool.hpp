/*=============================================================================
    Copyright (c) 2003 Joel de Guzman
    Copyright (c) 2003-2012 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/
#ifndef OCLM_UTIL_SAFE_BOOL_HPP
#define OCLM_UTIL_SAFE_BOOL_HPP

namespace oclm { namespace util
{
    template<typename Tag>
    class safe_bool
    {
    public:
        typedef void (safe_bool::*result_type)();
        result_type operator()(bool b) const
        {
            return b ? &safe_bool::true_ : 0;
        }

    private:
        void true_() {}
    };

}}

#endif

