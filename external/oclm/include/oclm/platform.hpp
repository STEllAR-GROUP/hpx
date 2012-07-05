
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_PLATFORM_HPP
#define OCLM_PLATFORM_HPP

#include <vector>
#include <string>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <boost/mpl/bool.hpp>

#include <boost/utility/enable_if.hpp>

#include <oclm/exception.hpp>
#include <oclm/device.hpp>
#include <oclm/platform_info.hpp>
#include <oclm/get_info.hpp>
#include <oclm/util/static.hpp>
#include <oclm/util/safe_bool.hpp>

#include <CL/cl.h>

namespace oclm
{

#define OCLM_DEVICE_INFO_LIMIT 12

    struct platform;

    struct platform
    {
        platform() : id_(0) {}

        explicit platform(cl_platform_id id)
            : devices(init_devices(id))
            , id_(id)
        {}

        platform& operator=(cl_platform_id id)
        {
            if(id_ != id)
            {
                id_ = id;
                devices = init_devices(id);
            }

            return *this;
        }

        template <typename Info>
        typename boost::enable_if<
            typename is_platform_info<Info>::type
          , typename Info::result_type
        >::type
        get(Info) const
        {
            return get_info<Info>(id_);
        }

        template <typename Info>
        typename boost::disable_if<
            typename is_platform_info<Info>::type
          , void
        >::type
        get(Info) const
        {
            static_assert(
                is_platform_info<Info>::value
              , "Template parameter is not a valid platform info type"
            );
        }

        operator cl_platform_id const &() const
        {
            return id_;
        }

        operator util::safe_bool<platform>::result_type() const
        {
            return util::safe_bool<platform>()(id_ != 0);
        }
            
        std::vector<device> devices;

        private:
            cl_platform_id id_;

            friend struct platform_manager;
            friend bool operator==(platform const & p1, platform const & p2);
    };

    inline bool operator==(platform const & p1, platform const & p2)
    {
        return p1.id_ == p2.id_;
    }

    struct platform_manager
        : boost::noncopyable
    {

        platform_manager();

        static HPX_EXPORT platform_manager & get();

        platform default_platform;
        device default_device;
        std::vector<platform> platforms;
    };

    //template <typename T>
    inline platform get_platform()//T const &)
    {
        //std::vector<platform> platforms = platform_manager::get().platforms;

        return platform_manager::get().default_platform;
    }

    //template <typename T>
    inline std::vector<platform> const & get_platforms()//T const &)
    {
        //std::vector<platform> platforms = platform_manager::get().platforms;

        return platform_manager::get().platforms;
    }
}

#endif
