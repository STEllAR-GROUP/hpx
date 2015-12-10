//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_CONTAINER_HPP
#define HPX_SERIALIZATION_CONTAINER_HPP

#include <hpx/config.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>
#include <hpx/runtime/serialization/basic_archive.hpp>
#include <hpx/util/assert.hpp>

namespace hpx { namespace serialization
{
    struct erased_output_container
    {
        virtual ~erased_output_container() {}

        virtual bool is_saving() const { return false; }
        virtual bool is_future_awaiting() const = 0;
        virtual void await_future(
            hpx::lcos::detail::future_data_refcnt_base & future_data) = 0;
        virtual void add_gid(
            naming::gid_type const & gid,
            naming::gid_type const & splitted_gid) = 0;
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void save_binary(void const* address, std::size_t count) = 0;
        virtual void save_binary_chunk(void const* address, std::size_t count) = 0;
    };

    struct erased_input_container
    {
        virtual ~erased_input_container() {}

        virtual bool is_saving() const { return false; }
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void load_binary(void * address, std::size_t count) = 0;
        virtual void load_binary_chunk(void * address, std::size_t count) = 0;
    };
}}

#endif
