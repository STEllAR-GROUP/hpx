
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_ASYNC_HPP
#define OCLM_ASYNC_HPP

#include <oclm/config.hpp>

#include <oclm/command_queue.hpp>
#include <oclm/packaged_kernel.hpp>

namespace oclm {

    template <typename Sig>
    event async(command_queue const & queue, packaged_kernel<Sig> pk)
    {
        std::size_t length[] = {pk.p_.content_.size()};
        const char* strings[] = {&pk.p_.content_[0]};
        cl_int err = CL_SUCCESS;
        cl_program p = clCreateProgramWithSource(queue.ctx_, 1, strings, length, &err);
        OCLM_THROW_IF_EXCEPTION(err, "clCreateProgramWithSource");

        // TODO: add error callback
        cl_device_id did = queue.d_;
        err = ::clBuildProgram(p, 1, &did, NULL, NULL, NULL);

        cl_kernel k = clCreateKernel(p, &pk.kernel_name_[0], &err);
        OCLM_THROW_IF_EXCEPTION(err, "clCreateKernel");

        std::vector<cl_event> events;

        pk.t0_.create(queue);
        pk.t0_.write(queue, events);
        pk.t1_.create(queue);
        pk.t1_.write(queue, events);
        pk.t2_.create(queue);
        pk.t2_.write(queue, events);

        pk.t0_.set_kernel_arg(k, 0);
        pk.t1_.set_kernel_arg(k, 1);
        pk.t2_.set_kernel_arg(k, 2);

        cl_event kernel_event;

        err = clEnqueueNDRangeKernel(
            queue
          , k
          , pk.ranges_.global_r.r.dim()
          , &pk.ranges_.offset_r.r.values[0]
          , &pk.ranges_.global_r.r.values[0]
          , &pk.ranges_.local_r.r.values[0]
          , events.size()
          , &events[0]
          , &kernel_event
        );
        OCLM_THROW_IF_EXCEPTION(err, "clEnqueueNDRangeKernel");

        {
            std::vector<cl_event> tmp; std::swap(events, tmp);
        }

        
        pk.t0_.read(queue, kernel_event, events);
        pk.t1_.read(queue, kernel_event, events);
        pk.t2_.read(queue, kernel_event, events);

        err = clWaitForEvents(events.size(), &events[0]);

        return event();
    }
}

#endif
