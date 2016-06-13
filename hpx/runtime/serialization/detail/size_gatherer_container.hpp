//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_SIZE_GATHERER_CONTAINER_MAY_27_2015_0818AM)
#define HPX_SERIALIZATION_SIZE_GATHERER_CONTAINER_MAY_27_2015_0818AM

// This 'container' is used to gather the required archive size for a given
// type before it is serialized.
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>

#include <cstddef>

namespace hpx { namespace serialization { namespace detail
{
    template <typename Container>
    struct access_data;

    class size_gatherer_container
    {
    public:
        size_gatherer_container() : size_(0) {}

        std::size_t size() const { return size_; }
        void resize(std::size_t size) { size_ = size; }

    private:
        std::size_t size_;
    };

    template <>
    struct access_data<size_gatherer_container>
    {
        static bool is_saving() { return false; }
        static bool is_future_awaiting() { return false; }

        static void await_future(
            size_gatherer_container& cont
          , hpx::lcos::detail::future_data_refcnt_base & future_data)
        {}

        static void add_gid(size_gatherer_container& cont,
                naming::gid_type const & gid,
                naming::gid_type const & splitted_gid)
        {}

        static void
        write(size_gatherer_container& cont, std::size_t count,
            std::size_t current, void const* address)
        {
        }

        static bool
        flush(binary_filter* filter, size_gatherer_container& cont,
            std::size_t current, std::size_t size, std::size_t written)
        {
            return true;
        }
    };
}}}

#endif
