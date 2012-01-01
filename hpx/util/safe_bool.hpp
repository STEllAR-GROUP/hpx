/*=============================================================================
    Copyright (c) 2003 Joel de Guzman
    Copyright (c) 2003-2012 Hartmut Kaiser

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
=============================================================================*/
#if !defined(HPX_UTIL_MAR_26_0200PM)
#define HPX_UTIL_MAR_26_0200PM

namespace hpx { namespace util
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

