
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_DEVICE_HPP
#define OCLM_DEVICE_HPP

#include <boost/mpl/bool.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/utility/enable_if.hpp>

#include <vector>
#include <string>

#include <oclm/exception.hpp>
#include <oclm/get_info.hpp>
#include <oclm/device_info.hpp>

#include <CL/cl.h>

namespace oclm
{
    struct device;

    std::vector<device> init_devices(cl_platform_id platform_id);
    device get_device();
    device create_subdevice(device const & d);

    struct device
    {
        static const device_type<CL_DEVICE_TYPE_ACCELERATOR> accelerator;
        static const device_type<CL_DEVICE_TYPE_ALL> all;
        static const device_type<CL_DEVICE_TYPE_CPU> cpu;
#ifdef CL_VERSION_1_2
        static const device_type<CL_DEVICE_TYPE_CUSTOM> custom;
#endif
        static const device_type<CL_DEVICE_TYPE_DEFAULT> default_;
        static const device_type<CL_DEVICE_TYPE_GPU> gpu;

        //FIXME: add reference counting

        device() {}

        explicit device(cl_device_id id)
            : id_(id)
        {
#ifdef CL_VERSION_1_2
            ::clRetainDevice(id_);
#endif
        }

        device& operator=(cl_device_id id)
        {
            if(id_ != id)
            {
                id_ = id;
#ifdef CL_VERSION_1_2
                ::clRetainDevice(id_);
#endif
            }

            return *this;
        }

        device(device const & d)
            : id_(d.id_)
        {
#ifdef CL_VERSION_1_2
            ::clRetainDevice(id_);
#endif
        }

        ~device()
        {
#ifdef CL_VERSION_1_2
            ::clReleaseDevice(id_);
#endif
        }

        template <typename Info>
        typename boost::enable_if<
            typename is_device_info<Info>::type
          , typename Info::result_type
        >::type
        get(Info) const
        {
            return get_info<Info>(id_);
        }

        template <typename Info>
        typename boost::disable_if<
            typename is_device_info<Info>::type
          , void
        >::type
        get(Info) const
        {
            static_assert(
                is_device_info<Info>::value
              , "Template parameter is not a valid device info type"
            );
        }
        
        operator cl_device_id const &() const
        {
            return id_;
        }

        private:
            cl_device_id id_;

            friend bool operator==(device const & d1, device const & d2);
    };

    inline bool operator==(device const & d1, device const & d2)
    {
        return d1.id_ == d2.id_;
    }

}

#endif
