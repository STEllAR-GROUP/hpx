
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_GET_INFO_HPP
#define OCLM_GET_INFO_HPP

#include <oclm/config.hpp>
#include <oclm/info.hpp>

namespace oclm
{
    namespace detail
    {
        template <typename Info, typename Result>
        struct get_info_helper
        {
            template <typename T>
            static Result call(T const & t)
            {
                Result r;
                cl_int err = CL_SUCCESS;
                err = Info::get(t, sizeof(Result), &r, NULL);
                OCLM_THROW_IF_EXCEPTION(err, "get_info");
                return r;
            }
        };
        
        template <typename Info, typename CharT, typename Allocator>
        struct get_info_helper<Info, std::basic_string<CharT, Allocator> >
        {
            template <typename T>
            static std::basic_string<CharT, Allocator> call(T const & t)
            {
                std::size_t n = 0;
                cl_int err = CL_SUCCESS;
                err = Info::get(t, 0, NULL, &n);
                OCLM_THROW_IF_EXCEPTION(err, "get_info");

                std::vector<char> buf(n);
                err = Info::get(t, n, &buf[0], NULL);
                OCLM_THROW_IF_EXCEPTION(err, "get_info");

                std::string ret(buf.begin(), buf.end());
                return ret;
            }
        };
        
        template <typename Info, typename VT, typename Allocator>
        struct get_info_helper<Info, std::vector<VT, Allocator> >
        {
            template <typename T>
            static std::vector<VT, Allocator> call(T const & t)
            {
                std::vector<VT, Allocator> res;

                std::size_t n = 0;
                cl_int err = CL_SUCCESS;
                err = Info::get(t, 0, NULL, &n);
                OCLM_THROW_IF_EXCEPTION(err, "get_info");

                res.resize(n/sizeof(VT));
                err = Info::get(t, n, &res[0], NULL);
                OCLM_THROW_IF_EXCEPTION(err, "get_info");

                return res;
            }
        };
    }

    template <typename Info, typename T>
    typename Info::result_type
    get_info(T const & t)
    {
        return detail::get_info_helper<Info, typename Info::result_type>::call(t);
    }
}

#endif
