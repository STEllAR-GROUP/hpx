//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/launch_policy.hpp

#if !defined(HPX_RUNTIME_LAUNCH_POLICY_AUG_13_2015_0647PM)
#define HPX_RUNTIME_LAUNCH_POLICY_AUG_13_2015_0647PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <type_traits>
#include <utility>

namespace hpx
{
    /// \cond NOINTERNAL
    namespace detail
    {
        enum class launch_policy
        {
            async = 0x01,
            deferred = 0x02,
            task = 0x04,  // see N3632
            sync = 0x08,
            fork = 0x10,  // same as async, but forces continuation stealing
            apply = 0x20,

            sync_policies = 0x0a,       // sync | deferred
            async_policies = 0x15,      // async | task | fork
            all = 0x3f                  // async | deferred | task | sync |
                                        // fork | apply
        };

        struct policy_holder_base
        {
            HPX_CONSTEXPR explicit policy_holder_base(launch_policy p,
                    threads::thread_priority priority =
                        threads::thread_priority_default) noexcept
              : policy_(p),
                priority_(priority)
            {}

            HPX_CONSTEXPR explicit operator bool() const noexcept
            {
                return is_valid();
            }

            HPX_CONSTEXPR launch_policy get_policy() const noexcept
            {
                return policy_;
            }

            HPX_CONSTEXPR bool is_valid() const noexcept
            {
                return static_cast<int>(policy_) != 0;
            }

            HPX_CONSTEXPR threads::thread_priority get_priority() const
            {
                return priority_;
            }

            launch_policy policy_;
            threads::thread_priority priority_;

        private:
            friend class serialization::access;

            HPX_EXPORT void load(serialization::input_archive& ar, unsigned);
            HPX_EXPORT void save(serialization::output_archive& ar, unsigned) const;

            HPX_SERIALIZATION_SPLIT_MEMBER()
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived = void>
        struct policy_holder : policy_holder_base
        {
            HPX_CONSTEXPR explicit policy_holder(launch_policy p,
                    threads::thread_priority priority =
                        threads::thread_priority_default) noexcept
              : policy_holder_base(p, priority)
            {}

            HPX_CONSTEXPR explicit policy_holder(policy_holder_base p) noexcept
              : policy_holder_base(p)
            {}

            HPX_CONSTEXPR operator launch_policy() const noexcept
            {
                return static_cast<Derived const*>(this)->get_policy();
            }

            HPX_CONSTEXPR explicit operator bool() const noexcept
            {
                return static_cast<Derived const*>(this)->is_valid();
            }

            HPX_CONSTEXPR launch_policy policy() const
            {
                return static_cast<Derived const*>(this)->get_policy();
            }
            HPX_CONSTEXPR threads::thread_priority priority() const
            {
                return static_cast<Derived const*>(this)->get_priority();
            }
        };

        template <>
        struct policy_holder<void> : policy_holder_base
        {
            HPX_CONSTEXPR explicit policy_holder(launch_policy p,
                    threads::thread_priority priority =
                        threads::thread_priority_default) noexcept
              : policy_holder_base(p, priority)
            {}

            HPX_CONSTEXPR explicit policy_holder(policy_holder_base p) noexcept
              : policy_holder_base(p)
            {}

            HPX_CONSTEXPR operator launch_policy() const noexcept
            {
                return this->policy_holder_base::get_policy();
            }

            HPX_CONSTEXPR explicit operator bool() const noexcept
            {
                return this->policy_holder_base::is_valid();
            }

            HPX_CONSTEXPR launch_policy policy() const
            {
                return this->policy_holder_base::get_policy();
            }
            HPX_CONSTEXPR threads::thread_priority priority() const
            {
                return this->policy_holder_base::get_priority();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        struct async_policy : policy_holder<async_policy>
        {
            HPX_CONSTEXPR explicit async_policy(
                    threads::thread_priority priority =
                        threads::thread_priority_default) noexcept
              : policy_holder<async_policy>(launch_policy::async, priority)
            {}

            HPX_CONSTEXPR async_policy operator()(
                threads::thread_priority priority) const noexcept
            {
                return async_policy(priority);
            }
        };

        struct fork_policy : policy_holder<fork_policy>
        {
            HPX_CONSTEXPR explicit fork_policy(
                    threads::thread_priority priority =
                        threads::thread_priority_boost) noexcept
              : policy_holder<fork_policy>(launch_policy::fork, priority)
            {}

            HPX_CONSTEXPR fork_policy operator()(
                threads::thread_priority priority) const noexcept
            {
                return fork_policy(priority);
            }
        };

        struct sync_policy : policy_holder<sync_policy>
        {
            HPX_CONSTEXPR sync_policy() noexcept
              : policy_holder<sync_policy>(launch_policy::sync)
            {}
        };

        struct deferred_policy : policy_holder<deferred_policy>
        {
            HPX_CONSTEXPR deferred_policy() noexcept
              : policy_holder<deferred_policy>(launch_policy::deferred)
            {}
        };

        struct apply_policy : policy_holder<apply_policy>
        {
            HPX_CONSTEXPR apply_policy() noexcept
              : policy_holder<apply_policy>(launch_policy::apply)
            {}
        };

        template <typename Pred>
        struct lazy_policy : policy_holder<lazy_policy<Pred> >
        {
            template <typename F, typename U =
                typename std::enable_if<!std::is_same<
                    lazy_policy<Pred>, typename std::decay<F>::type
                >::value>::type>
            explicit lazy_policy(F && f, threads::thread_priority priority =
                        threads::thread_priority_default)
              : policy_holder<lazy_policy<Pred> >(launch_policy::async, priority)
              , pred_(std::forward<F>(f))
            {}

            HPX_CONSTEXPR launch_policy get_policy() const
            {
                return pred_();
            }

            HPX_CONSTEXPR bool is_valid() const noexcept
            {
                return true;
            }

        private:
            Pred pred_;
        };

        struct lazy_policy_generator
        {
            HPX_CONSTEXPR async_policy operator()(
                threads::thread_priority priority) const noexcept
            {
                return async_policy(priority);
            }

            template <typename F>
            lazy_policy<typename std::decay<F>::type> operator()(F && f,
                threads::thread_priority priority =
                    threads::thread_priority_default) const
            {
                return lazy_policy<typename std::decay<F>::type>(
                    std::forward<F>(f), priority);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Left, typename Right>
        HPX_CONSTEXPR inline policy_holder_base operator&(
            policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return policy_holder_base(
                static_cast<launch_policy>(
                    static_cast<int>(lhs.policy()) &
                        static_cast<int>(rhs.policy())
                ));
        }

        template <typename Left, typename Right>
        HPX_CONSTEXPR inline policy_holder_base operator|(
            policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return policy_holder_base(
                static_cast<launch_policy>(
                    static_cast<int>(lhs.policy()) |
                        static_cast<int>(rhs.policy())
                ));
        }

        template <typename Left, typename Right>
        HPX_CONSTEXPR inline policy_holder_base operator^(
            policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return policy_holder_base(
                static_cast<launch_policy>(
                    static_cast<int>(lhs.policy()) ^
                        static_cast<int>(rhs.policy())
                ));
        }

        template <typename Derived>
        HPX_CONSTEXPR inline policy_holder<Derived>
        operator~(policy_holder<Derived> const& p) noexcept
        {
            return policy_holder<Derived>(
                static_cast<launch_policy>(
                    ~static_cast<int>(p.policy())
                ));
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
        HPX_CONSTEXPR inline bool operator==(policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return static_cast<int>(lhs.policy()) == static_cast<int>(rhs.policy());
        }

        template <typename Left, typename Right>
        HPX_CONSTEXPR inline bool operator!=(policy_holder<Left> const& lhs,
            policy_holder<Right> const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    /// Launch policies for \a hpx::async etc.
    struct launch : detail::policy_holder<>
    {
        ///////////////////////////////////////////////////////////////////////
        /// Default constructor. This creates a launch policy representing all
        /// possible launch modes
        HPX_CONSTEXPR launch() noexcept
          : detail::policy_holder<>{detail::launch_policy::all}
        {}

        /// \cond NOINTERNAL
        template <typename Derived>
        HPX_CONSTEXPR launch(detail::policy_holder<Derived> const& ph) noexcept
          : detail::policy_holder<>{ph}
        {}

        HPX_CONSTEXPR launch(detail::policy_holder_base const& ph) noexcept
          : detail::policy_holder<>{ph}
        {}
        /// \endcond

        /// Create a launch policy representing asynchronous execution
        HPX_CONSTEXPR launch(detail::async_policy) noexcept
          : detail::policy_holder<>{detail::launch_policy::async}
        {}

        /// Create a launch policy representing asynchronous execution. The
        /// new thread is executed in a preferred way
        HPX_CONSTEXPR launch(detail::fork_policy) noexcept
          : detail::policy_holder<>{detail::launch_policy::fork}
        {}

        /// Create a launch policy representing synchronous execution
        HPX_CONSTEXPR launch(detail::sync_policy) noexcept
          : detail::policy_holder<>{detail::launch_policy::sync}
        {}

        /// Create a launch policy representing deferred execution
        HPX_CONSTEXPR launch(detail::deferred_policy) noexcept
          : detail::policy_holder<>{detail::launch_policy::deferred}
        {}

        /// Create a launch policy representing fire and forget execution
        HPX_CONSTEXPR launch(detail::apply_policy) noexcept
          : detail::policy_holder<>{detail::launch_policy::apply}
        {}

        /// Create a launch policy representing fire and forget execution
        template <typename F>
        HPX_CONSTEXPR launch(detail::lazy_policy<F> const& p) noexcept
          : detail::policy_holder<>{p.policy()}
        {}

        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL
        using async_policy = detail::async_policy;
        using fork_policy = detail::fork_policy;
        using sync_policy = detail::sync_policy;
        using deferred_policy = detail::deferred_policy;
        using apply_policy = detail::apply_policy;
        template <typename F>
        using lazy_policy = detail::lazy_policy<F>;
        /// \endcond

        ///////////////////////////////////////////////////////////////////////
        /// Predefined launch policy representing asynchronous execution
        HPX_EXPORT static const detail::async_policy async;

        /// Predefined launch policy representing asynchronous execution.The
        /// new thread is executed in a preferred way
        HPX_EXPORT static const detail::fork_policy fork;

        /// Predefined launch policy representing synchronous execution
        HPX_EXPORT static const detail::sync_policy sync;

        /// Predefined launch policy representing deferred execution
        HPX_EXPORT static const detail::deferred_policy deferred;

        /// Predefined launch policy representing fire and forget execution
        HPX_EXPORT static const detail::apply_policy apply;

        /// Predefined launch policy representing conditionally asynchronous
        /// execution
        HPX_EXPORT static const detail::lazy_policy_generator lazy;

        /// \cond NOINTERNAL
        HPX_EXPORT static const detail::policy_holder<> all;
        HPX_EXPORT static const detail::policy_holder<> sync_policies;
        HPX_EXPORT static const detail::policy_holder<> async_policies;
        /// \endcond
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_FORCEINLINE HPX_CONSTEXPR
        bool has_async_policy(launch p) noexcept
        {
            return bool(
                static_cast<int>(p.get_policy()) &
                    static_cast<int>(detail::launch_policy::async_policies)
            );
        }

        template <typename F>
        HPX_FORCEINLINE HPX_CONSTEXPR
        bool has_async_policy(detail::policy_holder<F> const& p) noexcept
        {
            return bool(
                static_cast<int>(p.policy()) &
                    static_cast<int>(detail::launch_policy::async_policies)
            );
        }
    }
    /// \endcond
}

#endif
