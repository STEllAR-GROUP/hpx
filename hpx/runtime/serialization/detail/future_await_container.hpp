//  Copyright (c) 2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SERIALIZATION_FUTURE_AWAIT_CONTAINER_HPP)
#define HPX_SERIALIZATION_FUTURE_AWAIT_CONTAINER_HPP

// This 'container' is used to gather futures that need to become
// ready before the actual serialization process can be started

#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/unwrapped.hpp>

#include <vector>
namespace hpx
{
    bool is_starting();
}

namespace hpx { namespace serialization { namespace detail
{
    template <typename Container>
    struct access_data;

    class future_await_container
    {
    public:
        future_await_container() {}

        std::size_t size() const { return 0; }
        void resize(std::size_t size) { }

        void await_future(hpx::future<void> && f)
        {
            if(f.is_ready()) return;
            futures_.push_back(std::move(f));
        }

        bool has_futures() const
        {
            return !futures_.empty();
        }

        template <typename F>
        void operator()(F f)
        {
//             HPX_ASSERT(!hpx::is_starting());
            hpx::lcos::local::dataflow(//hpx::launch::sync,
                util::unwrapped(std::move(f)), futures_);
        }

    private:
        std::vector<future<void>> futures_;
    };

    template <>
    struct access_data<future_await_container>
    {
        static bool is_saving() { return false; }
        static bool is_future_awaiting() { return true; }

        static void await_future(future_await_container& cont, hpx::future<void> && f)
        {
            cont.await_future(std::move(f));
        }

        static void
        write(future_await_container& cont, std::size_t count,
            std::size_t current, void const* address)
        {
        }

        static bool
        flush(binary_filter* filter, future_await_container& cont,
            std::size_t current, std::size_t size, std::size_t written)
        {
            return true;
        }
    };
}}}

#endif
