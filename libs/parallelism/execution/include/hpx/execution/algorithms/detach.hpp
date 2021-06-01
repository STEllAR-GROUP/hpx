//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/allocator_deleter.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/thread_support/atomic_count.hpp>

#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    namespace detail {
        template <typename S, typename Allocator>
        struct operation_state_holder
        {
            struct detach_receiver
            {
                hpx::intrusive_ptr<operation_state_holder> os;

                template <typename E>
                    HPX_NORETURN void set_error(E&&) && noexcept
                {
                    HPX_ASSERT_MSG(false,
                        "set_error was called on the receiver of detach, "
                        "terminating. If you want to allow errors from the "
                        "predecessor sender, handle them first with e.g. "
                        "let_error.");
                    std::terminate();
                }

                void set_done() && noexcept
                {
                    os.reset();
                };

                template <typename... Ts>
                    void set_value(Ts&&...) && noexcept
                {
                    os.reset();
                }
            };

        private:
            using allocator_type = typename std::allocator_traits<
                Allocator>::template rebind_alloc<operation_state_holder>;
            allocator_type alloc;
            hpx::util::atomic_count count{0};

            using operation_state_type = connect_result_t<S, detach_receiver>;
            std::decay_t<operation_state_type> os;

        public:
            template <typename S_,
                typename = std::enable_if_t<!std::is_same<std::decay_t<S_>,
                    operation_state_holder>::value>>
            explicit operation_state_holder(S_&& s, allocator_type const& alloc)
              : alloc(alloc)
              , os(connect(std::forward<S_>(s), detach_receiver{this}))
            {
                start(os);
            }

        private:
            friend void intrusive_ptr_add_ref(operation_state_holder* p)
            {
                ++p->count;
            }

            friend void intrusive_ptr_release(operation_state_holder* p)
            {
                if (--p->count == 0)
                {
                    allocator_type other_alloc(p->alloc);
                    std::allocator_traits<allocator_type>::destroy(
                        other_alloc, p);
                    std::allocator_traits<allocator_type>::deallocate(
                        other_alloc, p, 1);
                }
            }
        };
    }    // namespace detail

    HPX_INLINE_CONSTEXPR_VARIABLE struct detach_t final
      : hpx::functional::tag_fallback<detach_t>
    {
    private:
        template <typename S,
            typename Allocator = hpx::util::internal_allocator<>>
        friend constexpr HPX_FORCEINLINE void tag_fallback_dispatch(
            detach_t, S&& s, Allocator&& a = Allocator{})
        {
            using allocator_type = Allocator;
            using operation_state_type =
                detail::operation_state_holder<S, Allocator>;
            using other_allocator = typename std::allocator_traits<
                allocator_type>::template rebind_alloc<operation_state_type>;
            using allocator_traits = std::allocator_traits<other_allocator>;
            using unique_ptr = std::unique_ptr<operation_state_type,
                util::allocator_deleter<other_allocator>>;

            other_allocator alloc(a);
            unique_ptr p(allocator_traits::allocate(alloc, 1),
                hpx::util::allocator_deleter<other_allocator>{alloc});

            new (p.get()) operation_state_type{std::forward<S>(s), alloc};
            p.release();
        }

        friend constexpr HPX_FORCEINLINE auto tag_fallback_dispatch(detach_t)
        {
            return detail::partial_algorithm<detach_t>{};
        }
    } detach{};
}}}    // namespace hpx::execution::experimental
