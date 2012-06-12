
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_COMMAND_QUEUE_HPP
#define OCLM_COMMAND_QUEUE_HPP

#include <oclm/config.hpp>
#include <oclm/device.hpp>
#include <oclm/context.hpp>

namespace oclm
{
    struct command_queue
    {
        command_queue()
            : cq_(0)
        {}

        ~command_queue()
        {
            if(cq_ != 0) ::clReleaseCommandQueue(cq_);
        }

        command_queue(command_queue const & other)
            : cq_(other.cq_)
        {
            if(cq_ != 0) ::clRetainCommandQueue(cq_);
        }
        
        command_queue & operator=(command_queue const & other)
        {
            if(cq_ != other.cq_)
            {
                cq_ = other.cq_;
                if(cq_ != 0) ::clRetainCommandQueue(cq_);
            }

            return *this;
        }

        static cl_command_queue create(cl_context ctx, cl_device_id d)
        {
            cl_int err = CL_SUCCESS;
            cl_command_queue cq
                = ::clCreateCommandQueue(
                    ctx
                  , d
                  , CL_QUEUE_PROFILING_ENABLE
                  , &err
                );

            OCLM_THROW_IF_EXCEPTION(err, "clCreateCommandQueue");

            return cq;
        }

        command_queue(context const & ctx, device const & d)
            : ctx_(ctx)
            , d_(d)
            , cq_(create(ctx, d))
        {
        }

        template <typename F>
        command_queue(F const & f)
        {
            std::vector<device> devices;
            make_selector(f)(devices);

            if(devices.size() == 0)
            {
                throw exception("command_queue::command_queue(selector): No devices found!");
            }

            ctx_ = create_context(devices[0]);
            d_ = devices[0];

            cq_ = create(ctx_, d_);
        }
        // TODO: add more ctors for constructing a queue with explicit context etc

        operator cl_command_queue const &() const
        {
            return cq_;
        }

        context ctx_;
        device d_;
        cl_command_queue cq_;
    };
}

#endif
