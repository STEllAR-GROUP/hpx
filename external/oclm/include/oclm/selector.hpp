
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_SELECTOR_HPP
#define OCLM_SELECTOR_HPP

#include <oclm/device.hpp>
#include <oclm/context.hpp>
#include <oclm/platform.hpp>

#include <boost/foreach.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>

namespace oclm
{
    template <typename F, typename Dummy = void>
    struct is_platform_selector : boost::mpl::false_ {};

    template <typename F, typename Dummy = void>
    struct is_device_selector : boost::mpl::false_ {};
    
    template <cl_device_type Type>
    struct is_device_selector<device_type<Type> >
        : boost::mpl::true_
    {};

    template <typename F>
    struct is_selector
        : boost::mpl::or_<
            typename is_device_selector<F>::type
          , typename is_platform_selector<F>::type
        >
    {};

    template <typename F>
    struct selector
    {
        typedef std::vector<platform> platforms_type;
        typedef std::vector<device> devices_type;

        typedef void result_type;

        selector(F f) : f_(f) {}

        // This is the entry point of our selecting platforms ...
        result_type operator()(platforms_type & platforms) const
        {
            select_platforms(platforms);
        }

        // This is the entry point of our selecting devices ...
        result_type operator()(devices_type & devices) const
        {
            std::vector<platform> platforms;
            select_platforms(platforms);
            select_devices(
                platforms
              , devices
            );
        }

        bool operator()(platform const & p, platforms_type & platforms) const
        {
            return select(p, platforms, typename is_platform_selector<F>::type());
        }

        bool operator()(device const & d, devices_type & devices) const
        {
            return select(d, devices, typename is_device_selector<F>::type());
        }

        template <typename T>
        bool
        select(T const & t, std::vector<T> & ts, boost::mpl::true_) const
        {
            return f_(t, ts);
        }

        template <typename T>
        bool
        select(T const &, std::vector<T> &, boost::mpl::false_) const
        {
            return false;
        }
        
        void select_platforms(platforms_type & platforms) const
        {
            platforms.reserve(get_platforms().size());

            BOOST_FOREACH(platform const & p, get_platforms())
            {
                std::cout << p.get(platform_name) << "\n";
                if(select(p, platforms, typename is_platform_selector<F>::type()))
                {
                    if(std::find(platforms.begin(), platforms.end(), p) == platforms.end())
                    {
                        platforms.push_back(p);
                    }
                }
            }
            
            if(platforms.size() == 0)
            {
                platforms = get_platforms();
            }
        }

        void select_devices(platforms_type & platforms, devices_type & devices) const
        {
            std::cout << "-----------------\n";
            BOOST_FOREACH(platform const & p, platforms)
            {
                std::cout << "selected platform: " << p.get(platform_name) << "\n";
                BOOST_FOREACH(device const & d, p.devices)
                {
                    std::cout << d.get(device_name) << "\n";
                    if(select(d, devices, typename is_device_selector<F>::type()))
                    {
                        if(std::find(devices.begin(), devices.end(), d) == devices.end())
                        {
                            devices.push_back(d);
                        }
                    }
                }
            }
            std::cout << "selected devices:\n";
            BOOST_FOREACH(device const & d, devices)
            {
                std::cout << d.get(device_name) << "\n";
            }
            std::cout << "-----------------\n";
        }

        F f_;
    };

    template <typename F>
    selector<F> const & make_selector(selector<F> const & f)
    {
        return f;
    }

    template <typename F>
    typename boost::enable_if<typename is_selector<F>::type, selector<F> >::type
    make_selector(F const & f)
    {
        return selector<F>(f);
    }

    template <typename F, typename Dummy>
    struct is_platform_selector<selector<F>, Dummy >
        : is_platform_selector<F>
    {};
    
    template <typename F, typename Dummy>
    struct is_device_selector<selector<F>, Dummy >
        : is_device_selector<F>
    {};

}

#endif
