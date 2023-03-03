//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/unwrap_ref.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    // 11.3.1. queryable concept [exec.queries.queryable]
    // 1. A query object is a customization point object
    //    ([customization.point.object]) that accepts as its first argument
    //    a queryable object, and for each such invocation that is valid,
    //    produces a value of the corresponding property of the object.
    // 2. Unless otherwise specified, given a queryable object e, a query
    //    object Q, and a pack of subexpressions args, the value returned
    //    by the expression Q(e, args...) is valid as long as e is valid.
    //
    //    Let e be an object of type E. The type E models queryable if for
    //    each callable object Q and a pack of subexpressions args, if
    //    requires { Q(e, args...) } is true then Q(e, args...) meets any
    //    semantic requirements imposed by Q.
    //
    // Reference implementation avoids using libstdc++'s object concepts
    // because they instantiate a lot of templates.
    template <typename T>
    inline constexpr bool is_queryable_v = std::is_destructible_v<T>;

    namespace detail {

        // checks whether tag_invoke(CPO, Args...) is contextually convertible
        // to bool, apply this meta function only for tag_invoke overloads
        template <typename CPO>
        struct contextually_convertible_to_bool
        {
            template <typename EnableTag, typename... Args>
            struct apply : std::true_type
            {
            };

            template <typename... Args>
            struct apply<hpx::functional::enable_tag_invoke_t, Args...>
              : std::is_invocable_r<bool,
                    hpx::functional::tag_t<hpx::functional::tag_invoke>, CPO,
                    Args...>
            {
            };
        };
    }    // namespace detail

    // 1. An execution environment contains state associated with the
    //    completion of an asynchronous operation. Every receiver has an
    //    associated execution environment, accessible with the get_env receiver
    //    query. The state of an execution environment is accessed with
    //    customization point objects. An execution environment may respond to
    //    any number of these environment queries.
    //
    // 2. An environment query is a customization point object that accepts
    //    as its first argument an execution environment. For an environment
    //    query EQ and an object e of type no_env, the expression EQ(e) shall be
    //    ill-formed.
    //
    namespace exec_envs {

        // no_env is a special environment used by the sender concept and by the
        // get_completion_signatures customization point when the user has
        // specified no environment argument.
        //
        //  [Note: A user may choose to not specify an environment in order
        //         to see if a sender knows its completion signatures
        //         independent of any particular execution environment.
        //  -- end note]
        struct no_env
        {
            using type = no_env;
            using id = no_env;

            template <typename Tag, typename Env>
            friend std::enable_if_t<std::is_same_v<no_env, std::decay_t<Env>>>
                tag_invoke(Tag, Env) = delete;
        };

        struct empty_env
        {
            using type = empty_env;
            using id = empty_env;
        };

        template <typename Tag, typename Value, typename BaseEnv = empty_env>
        struct env
        {
            struct type
            {
                using id = env;

                HPX_NO_UNIQUE_ADDRESS util::unwrap_reference_t<Value> value_;
                HPX_NO_UNIQUE_ADDRESS BaseEnv base_env_{};

                // Forward only the receiver queries
                template <typename Tag2, typename... Args,
                    typename = std::enable_if_t<functional::is_tag_invocable_v<
                        Tag2, BaseEnv const&, Args...>>>
                friend constexpr auto tag_invoke(
                    Tag2 tag, type const& self, Args&&... args) noexcept
                    -> functional::tag_invoke_result_t<Tag2, BaseEnv const&,
                        Args...>
                {
                    return HPX_FORWARD(Tag2, tag)(
                        self.base_env_, HPX_FORWARD(Args, args)...);
                }

                template <typename... Args>
                friend constexpr auto
                tag_invoke(Tag, type const& self, Args&&...) noexcept(
                    std::is_nothrow_copy_constructible_v<
                        util::unwrap_reference_t<Value>>)
                    -> util::unwrap_reference_t<Value>
                {
                    return self.value_;
                }
            };
        };

        template <typename Tag, typename Value, typename BaseEnv = empty_env>
        using env_t = std::decay_t<hpx::meta::type<env<Tag, Value, BaseEnv>>>;

        template <typename Tag>
        struct make_env_t
        {
            template <typename Value>
            constexpr auto operator()(Value&& value) const
                noexcept(std::is_nothrow_copy_constructible_v<
                    util::unwrap_reference_t<std::decay_t<Value>>>)
                    -> env_t<Tag, std::decay_t<Value>>
            {
                return {HPX_FORWARD(Value, value)};
            }

            template <typename Value, typename BaseEnvId>
            constexpr auto operator()(Value&& value, BaseEnvId&& base_env) const
                -> env_t<Tag, std::decay_t<Value>, std::decay_t<BaseEnvId>>
            {
                return {HPX_FORWARD(Value, value),
                    HPX_FORWARD(BaseEnvId, base_env)};
            }
        };

        // For making an evaluation environment from a key/value pair, and
        // optionally another environment.
        template <typename Tag>
        inline constexpr exec_envs::make_env_t<Tag> make_env{};

    }    // namespace exec_envs

    using exec_envs::empty_env;
    using exec_envs::env;
    using exec_envs::make_env;
    using exec_envs::no_env;

    template <typename Env>
    struct is_no_env : std::is_same<std::decay_t<Env>, no_env>
    {
    };

    template <typename Env>
    inline constexpr bool is_no_env_v = is_no_env<Env>::value;

    // get_env is a customization point object. For some subexpression r,
    // get_env(r) is expression-equivalent to:
    //
    // 1. tag_invoke(execution::get_env, r) if that expression is well-formed.
    // 2. Otherwise, empty_env{}.
    //
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct get_env_t final
      : hpx::functional::detail::tag_fallback<get_env_t>
    {
    private:
        template <typename EnvProvider>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            get_env_t, EnvProvider&&) noexcept
        {
            return empty_env{};
        }
    } get_env{};

    template <typename T>
    using env_of_t = decltype(get_env(std::declval<T>()));

    // different versions of clang-format disagree
    // clang-format off
    template <typename Tag, typename Value, typename BaseEnv = empty_env>
    using make_env_t = decltype(
        make_env<Tag>(std::declval<Value&&>(), std::declval<BaseEnv&&>()));
    // clang-format on

    // execution::forwarding_env_query is used to ask a customization point
    // object whether it is an environment query that should be forwarded
    // through environment adaptors.
    //
    // The name execution::forwarding_env_query denotes a customization point
    // object. For some subexpression t, execution::forwarding_env_query(t) is
    // expression equivalent to:
    //
    // 1. tag_invoke(execution::forwarding_env_query, t), contextually converted
    //    to bool, if the tag_invoke expression is well formed.
    //
    //      - Mandates: The tag_invoke expression is indeed contextually
    //        convertible to bool, that expression and the contextual conversion
    //        are not potentially-throwing and are core constant expressions if
    //        t is a core constant expression.
    //
    // 2. Otherwise, false.
    //
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct forwarding_env_query_t
        final
      : hpx::functional::detail::tag_fallback_noexcept<forwarding_env_query_t,
            detail::contextually_convertible_to_bool<forwarding_env_query_t>>
    {
    private:
        template <typename T>
        friend constexpr HPX_FORCEINLINE auto tag_fallback_invoke(
            forwarding_env_query_t, T&&) noexcept
        {
            return false;
        }
    } forwarding_env_query{};

    template <typename T>
    inline constexpr bool is_environment_provider_v =
        std::is_same_v<T, hpx::util::invoke_result_t<get_env_t, T>>;
}    // namespace hpx::execution::experimental
