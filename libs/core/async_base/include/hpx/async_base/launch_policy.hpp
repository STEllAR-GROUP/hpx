//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/async_base/launch_policy

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/scheduling_properties.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace hpx {
    /// \cond NOINTERNAL
    namespace detail {

        enum class launch_policy : std::int8_t
        {
            async = 0x01,
            deferred = 0x02,
            task = 0x04,    // see N3632
            sync = 0x08,
            fork = 0x10,    // same as async, but forces continuation stealing
            apply = 0x20,

            sync_policies = 0x0a,     // sync | deferred
            async_policies = 0x15,    // async | task | fork
            all = 0x3f                // async | deferred | task | sync |
                                      // fork | apply
        };

        struct policy_holder_base
        {
            constexpr explicit policy_holder_base(launch_policy p,
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_(p)
              , priority_(priority)
              , stacksize_(stacksize)
              , hint_(hint)
            {
            }

            constexpr explicit operator bool() const noexcept
            {
                return is_valid();
            }

            constexpr launch_policy get_policy() const noexcept
            {
                return policy_;
            }

            constexpr bool is_valid() const noexcept
            {
                return static_cast<int>(policy_) != 0;
            }

            constexpr threads::thread_priority get_priority() const noexcept
            {
                return priority_;
            }

            constexpr threads::thread_stacksize get_stacksize() const noexcept
            {
                return stacksize_;
            }

            constexpr threads::thread_schedule_hint get_hint() const noexcept
            {
                return hint_;
            }

            void set_priority(threads::thread_priority priority) noexcept
            {
                priority_ = priority;
            }

            void set_stacksize(threads::thread_stacksize stacksize) noexcept
            {
                stacksize_ = stacksize;
            }

            void set_hint(threads::thread_schedule_hint hint) noexcept
            {
                hint_ = hint;
            }

        protected:
            launch_policy policy_;
            threads::thread_priority priority_;
            threads::thread_stacksize stacksize_;
            threads::thread_schedule_hint hint_;

        private:
            friend class serialization::access;

            HPX_CORE_EXPORT void load(
                serialization::input_archive& ar, unsigned);
            HPX_CORE_EXPORT void save(
                serialization::output_archive& ar, unsigned) const;

            HPX_SERIALIZATION_SPLIT_MEMBER()
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived = void>
        struct policy_holder : policy_holder_base
        {
            constexpr explicit policy_holder(launch_policy p,
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_holder_base(p, priority, stacksize, hint)
            {
            }

            constexpr explicit policy_holder(policy_holder_base p) noexcept
              : policy_holder_base(p)
            {
            }

            constexpr operator launch_policy() const noexcept
            {
                return static_cast<Derived const*>(this)->get_policy();
            }

            constexpr explicit operator bool() const noexcept
            {
                return static_cast<Derived const*>(this)->is_valid();
            }

            constexpr launch_policy policy() const noexcept
            {
                return static_cast<Derived const*>(this)->get_policy();
            }
            constexpr threads::thread_priority priority() const noexcept
            {
                return static_cast<Derived const*>(this)->get_priority();
            }
            constexpr threads::thread_stacksize stacksize() const noexcept
            {
                return static_cast<Derived const*>(this)->get_stacksize();
            }
            constexpr threads::thread_schedule_hint hint() const noexcept
            {
                return static_cast<Derived const*>(this)->get_hint();
            }
        };

        template <>
        struct policy_holder<void> : policy_holder_base
        {
            constexpr explicit policy_holder(launch_policy p,
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_holder_base(p, priority, stacksize, hint)
            {
            }

            constexpr explicit policy_holder(policy_holder_base p) noexcept
              : policy_holder_base(p)
            {
            }

            constexpr operator launch_policy() const noexcept
            {
                return this->policy_holder_base::get_policy();
            }

            constexpr explicit operator bool() const noexcept
            {
                return this->policy_holder_base::is_valid();
            }

            constexpr launch_policy policy() const noexcept
            {
                return this->policy_holder_base::get_policy();
            }
            constexpr threads::thread_priority priority() const noexcept
            {
                return this->policy_holder_base::get_priority();
            }
            constexpr threads::thread_stacksize stacksize() const noexcept
            {
                return this->policy_holder_base::get_stacksize();
            }
            constexpr threads::thread_schedule_hint hint() const noexcept
            {
                return this->policy_holder_base::get_hint();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct async_policy : policy_holder<async_policy>
        {
            constexpr explicit async_policy(
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_holder<async_policy>(
                    launch_policy::async, priority, stacksize, hint)
            {
            }

            friend async_policy tag_invoke(
                hpx::execution::experimental::with_priority_t,
                async_policy policy, threads::thread_priority priority) noexcept
            {
                auto policy_with_priority = policy;
                policy_with_priority.set_priority(priority);
                return policy_with_priority;
            }

            friend constexpr hpx::threads::thread_priority tag_invoke(
                hpx::execution::experimental::get_priority_t,
                async_policy policy) noexcept
            {
                return policy.priority();
            }

            friend async_policy tag_invoke(
                hpx::execution::experimental::with_stacksize_t,
                async_policy policy,
                threads::thread_stacksize stacksize) noexcept
            {
                auto policy_with_stacksize = policy;
                policy_with_stacksize.set_stacksize(stacksize);
                return policy_with_stacksize;
            }

            friend constexpr hpx::threads::thread_stacksize tag_invoke(
                hpx::execution::experimental::get_stacksize_t,
                async_policy policy) noexcept
            {
                return policy.stacksize();
            }

            friend async_policy tag_invoke(
                hpx::execution::experimental::with_hint_t, async_policy policy,
                threads::thread_schedule_hint hint) noexcept
            {
                auto policy_with_hint = policy;
                policy_with_hint.set_hint(hint);
                return policy_with_hint;
            }

            friend constexpr hpx::threads::thread_schedule_hint tag_invoke(
                hpx::execution::experimental::get_hint_t,
                async_policy policy) noexcept
            {
                return policy.hint();
            }
        };

        struct fork_policy : policy_holder<fork_policy>
        {
            constexpr explicit fork_policy(threads::thread_priority priority =
                                               threads::thread_priority::boost,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_holder<fork_policy>(
                    launch_policy::fork, priority, stacksize, hint)
            {
            }

            friend fork_policy tag_invoke(
                hpx::execution::experimental::with_priority_t,
                fork_policy policy, threads::thread_priority priority) noexcept
            {
                auto policy_with_priority = policy;
                policy_with_priority.set_priority(priority);
                return policy_with_priority;
            }

            friend constexpr hpx::threads::thread_priority tag_invoke(
                hpx::execution::experimental::get_priority_t,
                fork_policy policy) noexcept
            {
                return policy.priority();
            }

            friend fork_policy tag_invoke(
                hpx::execution::experimental::with_stacksize_t,
                fork_policy policy,
                threads::thread_stacksize stacksize) noexcept
            {
                auto policy_with_stacksize = policy;
                policy_with_stacksize.set_stacksize(stacksize);
                return policy_with_stacksize;
            }

            friend constexpr hpx::threads::thread_stacksize tag_invoke(
                hpx::execution::experimental::get_stacksize_t,
                fork_policy policy) noexcept
            {
                return policy.stacksize();
            }

            friend fork_policy tag_invoke(
                hpx::execution::experimental::with_hint_t, fork_policy policy,
                threads::thread_schedule_hint hint) noexcept
            {
                auto policy_with_hint = policy;
                policy_with_hint.set_hint(hint);
                return policy_with_hint;
            }

            friend constexpr hpx::threads::thread_schedule_hint tag_invoke(
                hpx::execution::experimental::get_hint_t,
                fork_policy policy) noexcept
            {
                return policy.hint();
            }
        };

        struct sync_policy : policy_holder<sync_policy>
        {
            constexpr explicit sync_policy(
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_holder<sync_policy>(
                    launch_policy::sync, priority, stacksize, hint)
            {
            }

            friend sync_policy tag_invoke(
                hpx::execution::experimental::with_priority_t,
                sync_policy policy, threads::thread_priority priority) noexcept
            {
                auto policy_with_priority = policy;
                policy_with_priority.set_priority(priority);
                return policy_with_priority;
            }

            friend constexpr hpx::threads::thread_priority tag_invoke(
                hpx::execution::experimental::get_priority_t,
                sync_policy policy) noexcept
            {
                return policy.priority();
            }

            friend sync_policy tag_invoke(
                hpx::execution::experimental::with_stacksize_t,
                sync_policy policy,
                threads::thread_stacksize stacksize) noexcept
            {
                auto policy_with_stacksize = policy;
                policy_with_stacksize.set_stacksize(stacksize);
                return policy_with_stacksize;
            }

            friend constexpr hpx::threads::thread_stacksize tag_invoke(
                hpx::execution::experimental::get_stacksize_t,
                sync_policy policy) noexcept
            {
                return policy.stacksize();
            }

            friend sync_policy tag_invoke(
                hpx::execution::experimental::with_hint_t, sync_policy policy,
                threads::thread_schedule_hint hint) noexcept
            {
                auto policy_with_hint = policy;
                policy_with_hint.set_hint(hint);
                return policy_with_hint;
            }

            friend constexpr hpx::threads::thread_schedule_hint tag_invoke(
                hpx::execution::experimental::get_hint_t,
                sync_policy policy) noexcept
            {
                return policy.hint();
            }
        };

        struct deferred_policy : policy_holder<deferred_policy>
        {
            constexpr explicit deferred_policy(
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_holder<deferred_policy>(
                    launch_policy::deferred, priority, stacksize, hint)
            {
            }

            friend deferred_policy tag_invoke(
                hpx::execution::experimental::with_priority_t,
                deferred_policy policy,
                threads::thread_priority priority) noexcept
            {
                auto policy_with_priority = policy;
                policy_with_priority.set_priority(priority);
                return policy_with_priority;
            }

            friend constexpr hpx::threads::thread_priority tag_invoke(
                hpx::execution::experimental::get_priority_t,
                deferred_policy policy) noexcept
            {
                return policy.priority();
            }

            friend deferred_policy tag_invoke(
                hpx::execution::experimental::with_stacksize_t,
                deferred_policy policy,
                threads::thread_stacksize stacksize) noexcept
            {
                auto policy_with_stacksize = policy;
                policy_with_stacksize.set_stacksize(stacksize);
                return policy_with_stacksize;
            }

            friend constexpr hpx::threads::thread_stacksize tag_invoke(
                hpx::execution::experimental::get_stacksize_t,
                deferred_policy policy) noexcept
            {
                return policy.stacksize();
            }

            friend deferred_policy tag_invoke(
                hpx::execution::experimental::with_hint_t,
                deferred_policy policy,
                threads::thread_schedule_hint hint) noexcept
            {
                auto policy_with_hint = policy;
                policy_with_hint.set_hint(hint);
                return policy_with_hint;
            }

            friend constexpr hpx::threads::thread_schedule_hint tag_invoke(
                hpx::execution::experimental::get_hint_t,
                deferred_policy policy) noexcept
            {
                return policy.hint();
            }
        };

        struct apply_policy : policy_holder<apply_policy>
        {
            constexpr explicit apply_policy(
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) noexcept
              : policy_holder<apply_policy>(
                    launch_policy::apply, priority, stacksize, hint)
            {
            }

            friend apply_policy tag_invoke(
                hpx::execution::experimental::with_priority_t,
                apply_policy policy, threads::thread_priority priority) noexcept
            {
                auto policy_with_priority = policy;
                policy_with_priority.set_priority(priority);
                return policy_with_priority;
            }

            friend constexpr hpx::threads::thread_priority tag_invoke(
                hpx::execution::experimental::get_priority_t,
                apply_policy policy) noexcept
            {
                return policy.priority();
            }

            friend apply_policy tag_invoke(
                hpx::execution::experimental::with_stacksize_t,
                apply_policy policy,
                threads::thread_stacksize stacksize) noexcept
            {
                auto policy_with_stacksize = policy;
                policy_with_stacksize.set_stacksize(stacksize);
                return policy_with_stacksize;
            }

            friend constexpr hpx::threads::thread_stacksize tag_invoke(
                hpx::execution::experimental::get_stacksize_t,
                apply_policy policy) noexcept
            {
                return policy.stacksize();
            }

            friend apply_policy tag_invoke(
                hpx::execution::experimental::with_hint_t, apply_policy policy,
                threads::thread_schedule_hint hint) noexcept
            {
                auto policy_with_hint = policy;
                policy_with_hint.set_hint(hint);
                return policy_with_hint;
            }

            friend constexpr hpx::threads::thread_schedule_hint tag_invoke(
                hpx::execution::experimental::get_hint_t,
                apply_policy policy) noexcept
            {
                return policy.hint();
            }
        };

        template <typename Pred>
        struct select_policy : policy_holder<select_policy<Pred>>
        {
            template <typename F,
                typename U = std::enable_if_t<
                    !std::is_same_v<select_policy<Pred>, std::decay_t<F>>>>
            explicit select_policy(F&& f,
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {})    // NOLINT
              : policy_holder<select_policy<Pred>>(
                    launch_policy::async, priority, stacksize, hint)
              , pred_(HPX_FORWARD(F, f))
            {
            }

            constexpr launch_policy get_policy() const
                noexcept(noexcept(std::declval<Pred>()()))
            {
                return pred_();
            }

            constexpr bool is_valid() const noexcept
            {
                return true;
            }

            friend select_policy tag_invoke(
                hpx::execution::experimental::with_priority_t,
                select_policy const& policy,
                threads::thread_priority priority) noexcept
            {
                auto policy_with_priority = policy;
                policy_with_priority.set_priority(priority);
                return policy_with_priority;
            }

            friend constexpr hpx::threads::thread_priority tag_invoke(
                hpx::execution::experimental::get_priority_t,
                select_policy const& policy) noexcept
            {
                return policy.priority();
            }

            friend select_policy tag_invoke(
                hpx::execution::experimental::with_stacksize_t,
                select_policy const& policy,
                threads::thread_stacksize stacksize) noexcept
            {
                auto policy_with_stacksize = policy;
                policy_with_stacksize.set_stacksize(stacksize);
                return policy_with_stacksize;
            }

            friend constexpr hpx::threads::thread_stacksize tag_invoke(
                hpx::execution::experimental::get_stacksize_t,
                select_policy const& policy) noexcept
            {
                return policy.stacksize();
            }

            friend select_policy tag_invoke(
                hpx::execution::experimental::with_hint_t,
                select_policy const& policy,
                threads::thread_schedule_hint hint) noexcept
            {
                auto policy_with_hint = policy;
                policy_with_hint.set_hint(hint);
                return policy_with_hint;
            }

            friend constexpr hpx::threads::thread_schedule_hint tag_invoke(
                hpx::execution::experimental::get_hint_t,
                select_policy const& policy) noexcept
            {
                return policy.hint();
            }

        private:
            Pred pred_;
        };

        struct select_policy_generator
        {
            constexpr async_policy operator()(threads::thread_priority priority,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) const noexcept
            {
                return async_policy(priority, stacksize, hint);
            }

            template <typename F>
            select_policy<std::decay_t<F>> operator()(F&& f,
                threads::thread_priority priority =
                    threads::thread_priority::default_,
                threads::thread_stacksize stacksize =
                    threads::thread_stacksize::default_,
                threads::thread_schedule_hint hint = {}) const
            {
                return select_policy<std::decay_t<F>>(
                    HPX_FORWARD(F, f), priority, stacksize, hint);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Left, typename Right>
        constexpr inline policy_holder_base operator&(
            policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return policy_holder_base(
                static_cast<launch_policy>(static_cast<int>(lhs.policy()) &
                    static_cast<int>(rhs.policy())),
                lhs.get_priority(), lhs.get_stacksize(), lhs.get_hint());
        }

        template <typename Left, typename Right>
        constexpr inline policy_holder_base operator|(
            policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return policy_holder_base(
                static_cast<launch_policy>(static_cast<int>(lhs.policy()) |
                    static_cast<int>(rhs.policy())),
                lhs.get_priority(), lhs.get_stacksize(), lhs.get_hint());
        }

        template <typename Left, typename Right>
        constexpr inline policy_holder_base operator^(
            policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return policy_holder_base(
                static_cast<launch_policy>(static_cast<int>(lhs.policy()) ^
                    static_cast<int>(rhs.policy())),
                lhs.get_priority(), lhs.get_stacksize(), lhs.get_hint());
        }

        template <typename Derived>
        constexpr inline policy_holder<Derived> operator~(
            policy_holder<Derived> const& p) noexcept
        {
            return policy_holder<Derived>(
                static_cast<launch_policy>(~static_cast<int>(p.policy())),
                p.get_priority(), p.get_stacksize(), p.get_hint());
        }

        template <typename Left, typename Right>
        inline policy_holder<Left> operator&=(
            policy_holder<Left>& lhs, policy_holder<Right> const& rhs) noexcept
        {
            lhs = policy_holder<Left>(lhs & rhs);
            return lhs;
        }

        template <typename Left, typename Right>
        inline policy_holder<Left> operator|=(
            policy_holder<Left>& lhs, policy_holder<Right> const& rhs) noexcept
        {
            lhs = policy_holder<Left>(lhs | rhs);
            return lhs;
        }

        template <typename Left, typename Right>
        inline policy_holder<Left> operator^=(
            policy_holder<Left>& lhs, policy_holder<Right> const& rhs) noexcept
        {
            lhs = policy_holder<Left>(lhs ^ rhs);
            return lhs;
        }

        template <typename Left, typename Right>
        constexpr inline bool operator==(policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return static_cast<int>(lhs.policy()) ==
                static_cast<int>(rhs.policy());
        }

        template <typename Left, typename Right>
        constexpr inline bool operator!=(policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
    }    // namespace detail
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// Launch policies for \a hpx::async etc.
    struct launch : detail::policy_holder<>
    {
        ///////////////////////////////////////////////////////////////////////
        /// Default constructor. This creates a launch policy representing all
        /// possible launch modes
        constexpr launch() noexcept
          : detail::policy_holder<>{detail::launch_policy::all}
        {
        }

        /// \cond NOINTERNAL
        template <typename Derived>
        constexpr launch(detail::policy_holder<Derived> const& ph) noexcept
          : detail::policy_holder<>{ph}
        {
        }

        constexpr launch(detail::policy_holder_base const& ph) noexcept
          : detail::policy_holder<>{ph}
        {
        }
        /// \endcond

        /// Create a launch policy representing asynchronous execution
        constexpr launch(detail::async_policy p) noexcept
          : detail::policy_holder<>{detail::launch_policy::async, p.priority(),
                p.stacksize(), p.hint()}
        {
        }

        /// Create a launch policy representing asynchronous execution. The
        /// new thread is executed in a preferred way
        constexpr launch(detail::fork_policy p) noexcept
          : detail::policy_holder<>{detail::launch_policy::fork, p.priority(),
                p.stacksize(), p.hint()}
        {
        }

        /// Create a launch policy representing synchronous execution
        constexpr launch(detail::sync_policy p) noexcept
          : detail::policy_holder<>{detail::launch_policy::sync, p.priority(),
                p.stacksize(), p.hint()}
        {
        }

        /// Create a launch policy representing deferred execution
        constexpr launch(detail::deferred_policy p) noexcept
          : detail::policy_holder<>{detail::launch_policy::deferred,
                p.priority(), p.stacksize(), p.hint()}
        {
        }

        /// Create a launch policy representing fire and forget execution
        constexpr launch(detail::apply_policy p) noexcept
          : detail::policy_holder<>{detail::launch_policy::apply, p.priority(),
                p.stacksize(), p.hint()}
        {
        }

        /// Create a launch policy representing fire and forget execution
        template <typename F>
        constexpr launch(detail::select_policy<F> const& p) noexcept
          : detail::policy_holder<>{
                p.policy(), p.priority(), p.stacksize(), p.hint()}
        {
        }

        template <typename Launch,
            typename Enable =
                std::enable_if_t<hpx::traits::is_launch_policy_v<Launch>>>
        constexpr launch(Launch l, threads::thread_priority priority,
            threads::thread_stacksize stacksize,
            threads::thread_schedule_hint hint) noexcept
          : detail::policy_holder<>(l.policy(), priority, stacksize, hint)
        {
        }

        ///////////////////////////////////////////////////////////////////////
        friend launch tag_invoke(hpx::execution::experimental::with_priority_t,
            launch const& policy, threads::thread_priority priority) noexcept
        {
            auto policy_with_priority = policy;
            policy_with_priority.set_priority(priority);
            return policy_with_priority;
        }

        friend constexpr hpx::threads::thread_priority tag_invoke(
            hpx::execution::experimental::get_priority_t,
            launch const& policy) noexcept
        {
            return policy.priority();
        }

        friend launch tag_invoke(hpx::execution::experimental::with_stacksize_t,
            launch const& policy, threads::thread_stacksize stacksize) noexcept
        {
            auto policy_with_stacksize = policy;
            policy_with_stacksize.set_stacksize(stacksize);
            return policy_with_stacksize;
        }

        friend constexpr hpx::threads::thread_stacksize tag_invoke(
            hpx::execution::experimental::get_stacksize_t,
            launch const& policy) noexcept
        {
            return policy.stacksize();
        }

        friend launch tag_invoke(hpx::execution::experimental::with_hint_t,
            launch const& policy, threads::thread_schedule_hint hint) noexcept
        {
            auto policy_with_hint = policy;
            policy_with_hint.set_hint(hint);
            return policy_with_hint;
        }

        friend constexpr hpx::threads::thread_schedule_hint tag_invoke(
            hpx::execution::experimental::get_hint_t,
            launch const& policy) noexcept
        {
            return policy.hint();
        }

        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL
        using async_policy = detail::async_policy;
        using fork_policy = detail::fork_policy;
        using sync_policy = detail::sync_policy;
        using deferred_policy = detail::deferred_policy;
        using apply_policy = detail::apply_policy;
        template <typename F>
        using select_policy = detail::select_policy<F>;
        /// \endcond

        ///////////////////////////////////////////////////////////////////////
        /// Predefined launch policy representing asynchronous execution
        HPX_CORE_EXPORT static const detail::async_policy async;

        /// Predefined launch policy representing asynchronous execution.The
        /// new thread is executed in a preferred way
        HPX_CORE_EXPORT static const detail::fork_policy fork;

        /// Predefined launch policy representing synchronous execution
        HPX_CORE_EXPORT static const detail::sync_policy sync;

        /// Predefined launch policy representing deferred execution
        HPX_CORE_EXPORT static const detail::deferred_policy deferred;

        /// Predefined launch policy representing fire and forget execution
        HPX_CORE_EXPORT static const detail::apply_policy apply;

        /// Predefined launch policy representing delayed policy selection
        HPX_CORE_EXPORT static const detail::select_policy_generator select;

        /// \cond NOINTERNAL
        HPX_CORE_EXPORT static const detail::policy_holder<> all;
        HPX_CORE_EXPORT static const detail::policy_holder<> sync_policies;
        HPX_CORE_EXPORT static const detail::policy_holder<> async_policies;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace detail {
        HPX_FORCEINLINE constexpr bool has_async_policy(launch p) noexcept
        {
            return bool(static_cast<int>(p.get_policy()) &
                static_cast<int>(detail::launch_policy::async_policies));
        }

        template <typename F>
        HPX_FORCEINLINE constexpr bool has_async_policy(
            detail::policy_holder<F> const& p) noexcept
        {
            return bool(static_cast<int>(p.policy()) &
                static_cast<int>(detail::launch_policy::async_policies));
        }
    }    // namespace detail
    /// \endcond
}    // namespace hpx
