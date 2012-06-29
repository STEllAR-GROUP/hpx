
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <oclm/device.hpp>
#include <oclm/platform.hpp>

namespace oclm
{
    const device_type<CL_DEVICE_TYPE_ACCELERATOR> device::accelerator =
        device_type<CL_DEVICE_TYPE_ACCELERATOR>();
    const device_type<CL_DEVICE_TYPE_ALL> device::all =
        device_type<CL_DEVICE_TYPE_ALL>();
    const device_type<CL_DEVICE_TYPE_CPU> device::cpu =
        device_type<CL_DEVICE_TYPE_CPU>();
#ifdef CL_VERSION_1_2
    const device_type<CL_DEVICE_TYPE_CUSTOM> device::custom =
        device_type<CL_DEVICE_TYPE_CUSTOM>();
#endif
    const device_type<CL_DEVICE_TYPE_DEFAULT> device::default_ =
        device_type<CL_DEVICE_TYPE_DEFAULT>();
    const device_type<CL_DEVICE_TYPE_GPU> device::gpu =
        device_type<CL_DEVICE_TYPE_GPU>();

    std::vector<device> init_devices(cl_platform_id platform_id)
    {
            cl_uint n = 0;
            cl_int err = CL_SUCCESS;
            err = ::clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &n);
            OCLM_THROW_IF_EXCEPTION(err, "clGetDeviceIDs");
            std::vector<cl_device_id> device_ids(n);
            err = ::clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, n, &device_ids[0], NULL);
            OCLM_THROW_IF_EXCEPTION(err, "clGetDeviceIDs");

            std::vector<device> devices(n);
            std::copy(device_ids.begin(), device_ids.end(), devices.begin());

            return devices;
    }

    device get_device()
    {
        return platform_manager::get().default_device;
    }

    device create_subdevice(device const &)
    {
        return device();
    }
}
