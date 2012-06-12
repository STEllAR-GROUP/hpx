
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_INFO_SELECTOR_HPP
#define OCLM_INFO_SELECTOR_HPP

#include <oclm/selector.hpp>
#include <oclm/platform_info.hpp>
#include <oclm/device_info.hpp>

#include <string>

#include <boost/regex.hpp>

namespace oclm
{
    namespace detail
    {
        template <typename Info>
        struct info_equal_selector
        {
            typename Info::result_type value;

            template <typename T>
            bool operator()(T const & t, std::vector<T> &) const
            {
                return t.get(Info()) == value;
            }
        };
        
        template <typename Info>
        struct info_regex_selector
        {
            std::string value;

            template <typename T>
            bool operator()(T const & t, std::vector<T> &) const
            {
                boost::regex re(value);
                return boost::regex_search(t.get(Info()), re);
            }
        };
    }

    template <typename Info>
    struct is_platform_selector<detail::info_equal_selector<Info> >
        : is_platform_info<Info>::type
    {};

    template <typename Info>
    struct is_platform_selector<detail::info_regex_selector<Info> >
        : is_platform_info<Info>::type
    {};

    template <typename Info>
    struct is_device_selector<detail::info_equal_selector<Info> >
        : is_device_info<Info>
    {};

    template <typename Info>
    struct is_device_selector<detail::info_regex_selector<Info> >
        : is_device_info<Info>
    {};

    template <typename Info>
    selector<detail::info_equal_selector<Info> >
    operator==(Info, typename Info::result_type const & s)
    {
        detail::info_equal_selector<Info> sel = {s};
        return selector<detail::info_equal_selector<Info> >(sel);
    }
    
    template <typename Info>
    selector<detail::info_regex_selector<Info> >
    operator%=(Info, std::string const & s)
    {
        detail::info_regex_selector<Info> sel = {s};
        return selector<detail::info_regex_selector<Info> >(sel);
    }
}

#endif
