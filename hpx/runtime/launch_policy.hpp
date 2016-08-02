//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/launch_policy.hpp

#if !defined(HPX_RUNTIME_LAUNCH_POLICY_AUG_13_2015_0647PM)
#define HPX_RUNTIME_LAUNCH_POLICY_AUG_13_2015_0647PM

#include <hpx/config.hpp>

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

        struct policy_holder
        {
            HPX_CONSTEXPR explicit policy_holder(launch_policy p) HPX_NOEXCEPT
              : policy_(p)
            {}

            HPX_CONSTEXPR operator launch_policy() const HPX_NOEXCEPT
            {
                return policy_;
            }

            HPX_CONSTEXPR explicit operator bool() const HPX_NOEXCEPT
            {
                return static_cast<int>(policy_) != 0;
            }

            launch_policy policy_;
        };

        struct async_policy : policy_holder
        {
            HPX_CONSTEXPR async_policy() HPX_NOEXCEPT
              : policy_holder(launch_policy::async)
            {}
        };

        struct fork_policy : policy_holder
        {
            HPX_CONSTEXPR fork_policy() HPX_NOEXCEPT
              : policy_holder(launch_policy::fork)
            {}
        };

        struct sync_policy : policy_holder
        {
            HPX_CONSTEXPR sync_policy() HPX_NOEXCEPT
              : policy_holder(launch_policy::sync)
            {}
        };

        struct deferred_policy : policy_holder
        {
            HPX_CONSTEXPR deferred_policy() HPX_NOEXCEPT
              : policy_holder(launch_policy::deferred)
            {}
        };

        struct apply_policy : policy_holder
        {
            HPX_CONSTEXPR apply_policy() HPX_NOEXCEPT
              : policy_holder(launch_policy::apply)
            {}
        };

        ///////////////////////////////////////////////////////////////////////////
        HPX_CONSTEXPR inline policy_holder
        operator&(policy_holder lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            return policy_holder(static_cast<launch_policy>(
                    static_cast<int>(lhs.policy_) & static_cast<int>(rhs.policy_)
                ));
        }

        HPX_CONSTEXPR inline policy_holder
        operator|(policy_holder lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            return policy_holder(static_cast<launch_policy>(
                    static_cast<int>(lhs.policy_) | static_cast<int>(rhs.policy_)
                ));
        }

        HPX_CONSTEXPR inline policy_holder
        operator^(policy_holder lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            return policy_holder(static_cast<launch_policy>(
                    static_cast<int>(lhs.policy_) ^ static_cast<int>(rhs.policy_)
                ));
        }

        HPX_CONSTEXPR inline policy_holder
        operator~(policy_holder p) HPX_NOEXCEPT
        {
            return policy_holder(static_cast<launch_policy>(
                    ~static_cast<int>(p.policy_)
                ));
        }

        inline policy_holder
        operator&=(policy_holder& lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            lhs = lhs & rhs;
            return lhs;
        }

        inline policy_holder
        operator|=(policy_holder& lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            lhs = lhs | rhs;
            return lhs;
        }

        inline policy_holder
        operator^=(policy_holder& lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            lhs = lhs ^ rhs;
            return lhs;
        }

        HPX_CONSTEXPR inline bool
        operator==(policy_holder lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            return static_cast<int>(lhs.policy_) == static_cast<int>(rhs.policy_);
        }

        HPX_CONSTEXPR inline bool
        operator!=(policy_holder lhs, policy_holder rhs) HPX_NOEXCEPT
        {
            return static_cast<int>(lhs.policy_) != static_cast<int>(rhs.policy_);
        }
    }
    /// \endcond

    ///////////////////////////////////////////////////////////////////////
    // Launch policies for \a hpx::async etc.
    struct launch : detail::policy_holder
    {
        HPX_CONSTEXPR launch() HPX_NOEXCEPT
          : detail::policy_holder{detail::launch_policy::all}
        {}

        HPX_CONSTEXPR launch(detail::policy_holder ph) HPX_NOEXCEPT
          : detail::policy_holder{ph}
        {}

        HPX_CONSTEXPR launch(detail::async_policy) HPX_NOEXCEPT
          : detail::policy_holder{detail::launch_policy::async}
        {}

        HPX_CONSTEXPR launch(detail::fork_policy) HPX_NOEXCEPT
          : detail::policy_holder{detail::launch_policy::fork}
        {}

        HPX_CONSTEXPR launch(detail::sync_policy) HPX_NOEXCEPT
          : detail::policy_holder{detail::launch_policy::sync}
        {}

        HPX_CONSTEXPR launch(detail::deferred_policy) HPX_NOEXCEPT
          : detail::policy_holder{detail::launch_policy::deferred}
        {}

        HPX_CONSTEXPR launch(detail::apply_policy) HPX_NOEXCEPT
          : detail::policy_holder{detail::launch_policy::apply}
        {}

        HPX_EXPORT static const detail::async_policy async;
        HPX_EXPORT static const detail::fork_policy fork;
        HPX_EXPORT static const detail::sync_policy sync;
        HPX_EXPORT static const detail::deferred_policy deferred;
        HPX_EXPORT static const detail::apply_policy apply;

        HPX_EXPORT static const detail::policy_holder all;
        HPX_EXPORT static const detail::policy_holder sync_policies;
        HPX_EXPORT static const detail::policy_holder async_policies;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    namespace detail
    {
        HPX_FORCEINLINE bool has_async_policy(launch p) HPX_NOEXCEPT
        {
            return bool(p & launch::async_policies);
        }
    }
    /// \endcond
}

#endif
