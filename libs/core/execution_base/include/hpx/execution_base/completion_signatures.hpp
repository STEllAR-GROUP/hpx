//  Copyright (c) 2022-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <type_traits>
#include <utility>

#if defined(HPX_HAVE_CXX20_COROUTINES)
#include <hpx/assert.hpp>
#include <hpx/execution_base/operation_state.hpp>
#include <hpx/execution_base/traits/coroutine_traits.hpp>
#include <hpx/type_support/coroutines_support.hpp>

#include <exception>
#include <system_error>
#endif

namespace hpx::execution::experimental {

    // empty_variant will be returned by execution::value_types_of_t and
    // execution::error_types_of_t if no signatures are provided.
    struct empty_variant
    {
        empty_variant() = delete;
    };

    namespace detail {

        // use this remove_cv_ref instead of std::decay to avoid
        // decaying function types, e.g. set_value_t() -> set_value_t(*)()
        template <typename T>
        struct remove_cv_ref
        {
            using type = std::remove_cv_t<std::remove_reference_t<T>>;
        };

        template <typename T>
        using remove_cv_ref_t = meta::type<remove_cv_ref<T>>;

        // clang-format off

        // If sizeof...(Ts) is greater than zero, variant-or-empty<Ts...> names
        // the type variant<Us...> where Us... is the pack decay_t<Ts>... with
        // duplicate types removed.
        template <template <typename...> typename Variant>
        struct decay_variant_or_empty
        {
            template <typename... Ts>
            using apply = meta::invoke<
                meta::if_<meta::bool_<sizeof...(Ts) != 0>,
                    meta::transform<
                        meta::func1<remove_cv_ref_t>,
                        meta::unique<meta::func<Variant>>>,
                    meta::constant<empty_variant>>,
                Ts...>;
        };

        template <template <typename...> typename Variant>
        struct decay_variant
        {
            template <typename... Ts>
            using apply = meta::invoke<
                meta::transform<
                    meta::func1<remove_cv_ref_t>,
                    meta::unique<meta::func<Variant>>>,
                Ts...>;
        };

        template <template <typename...> typename Variant>
        struct unique_variant
        {
            template <typename... Ts>
            using apply =
                meta::invoke<meta::unique<meta::func<Variant>>, Ts...>;
        };
        // clang-format on

        template <typename... Ts>
        using decayed_variant =
            meta::invoke<decay_variant_or_empty<hpx::variant>, Ts...>;

        template <template <typename...> typename Tuple>
        struct decay_tuple
        {
            template <typename... Ts>
            using apply = Tuple<remove_cv_ref_t<Ts>...>;
        };

        template <typename... Ts>
        using decayed_tuple = meta::invoke<decay_tuple<hpx::tuple>, Ts...>;

        // test, if set_value_t(Ts...), set_error_t(Error, Ts...), or
        // set_stopped_t() are available, return meta::pack<Ts...>
        template <typename Tag, typename MetaF = meta::func<meta::pack>,
            typename... Ts>
        std::enable_if_t<std::is_same_v<Tag, set_value_t>,
            meta::pack<meta::invoke<MetaF, Ts...>>>
            test_signature(Tag (*)(Ts...));

        template <typename Tag, typename MetaF = meta::func<meta::pack>,
            typename Error, typename... Ts>
        std::enable_if_t<std::is_same_v<Tag, set_error_t>,
            meta::pack<meta::invoke1<MetaF, Error>,
                meta::invoke1<MetaF, Ts>...>>
            test_signature(Tag (*)(Error, Ts...));

        template <typename Tag, typename MetaF = meta::func<meta::pack>>
        std::enable_if_t<std::is_same_v<Tag, set_stopped_t>,
            meta::pack<meta::invoke<MetaF>>>
            test_signature(Tag (*)());

        // fallback, returns an empty pack<>
        template <typename, typename = void>
        meta::pack<> test_signature(...);

        // clang-format off
        template <typename Signature, typename Tag,
            typename MetaF = meta::func<meta::pack>>
        using signature_arg_apply = decltype(
            test_signature<Tag, MetaF>(static_cast<Signature*>(nullptr)));
        // clang-format on

        template <typename Signature, typename Enable = void>
        struct is_completion_signature : std::false_type
        {
        };

        template <typename Signature>
        struct is_completion_signature<Signature,
            std::void_t<decltype(test_signature(
                static_cast<Signature*>(nullptr)))>> : std::true_type
        {
        };

        template <typename... Signatures>
        struct compose_signatures
        {
            struct type
            {
                // clang-format off
                template <typename Tuple, typename Variant>
                struct make_value_types
                {
                    using type = meta::apply<Variant,
                        signature_arg_apply<
                            Signatures, set_value_t, Tuple
                        >...
                    >;
                };

                template <typename Variant>
                struct make_error_types
                {
                    using type = meta::apply<Variant,
                        signature_arg_apply<
                            Signatures, set_error_t,
                            meta::func1<meta::identity>
                        >...
                    >;
                };

                static constexpr bool sends_stopped =
                    meta::apply<meta::count,
                        signature_arg_apply<
                            Signatures, set_stopped_t
                        >...
                    >::value != 0;
                // clang-format on

                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = meta::type<
                    make_value_types<meta::func<Tuple>, meta::func<Variant>>>;

                template <template <typename...> typename Variant>
                using error_types =
                    meta::type<make_error_types<meta::func<Variant>>>;

                using signatures_t = meta::pack<Signatures...>;
            };
        };

        template <typename Pack, typename Enable = void>
        struct generate_completion_signatures
        {
        };

        template <typename... Signatures>
        struct generate_completion_signatures<meta::pack<Signatures...>,
            std::enable_if_t<
                util::all_of_v<is_completion_signature<Signatures>...>>>
          : meta::invoke<meta::func<compose_signatures>, Signatures...>
        {
        };
    }    // namespace detail

    // A type Fn satisfies completion-signature if it is a function type with
    // one of the following forms:
    //
    //      set_value_t(Vs...), where Vs is an arbitrary parameter pack.
    //
    //      set_error_t(E), where E is an arbitrary type.
    //
    //      set_stopped_t()
    //
    // Otherwise, Fn does not satisfy completion-signature.
    //
    // Let ValueFns be a template parameter pack of the function types in Fns
    // whose return types are execution::set_value_t, and let Valuesn be a
    // template parameter pack of the function argument types in the n-th type
    // in ValueFns.
    // Then, given two variadic templates Tuple and Variant, the type
    // completion_signatures<Fns...>::value_types<Tuple, Variant> names the type
    // Variant<Tuple<Values0...>, Tuple<Values1...>, ... Tuple<Valuesn-1...>>,
    // where m is the size of the parameter pack ValueFns.
    //
    // Let ErrorFns be a template parameter pack of the function types in Fns
    // whose return types are execution::set_error_t, and let Errorn be the
    // function argument type in the n-th type in ErrorFns.
    // Then, given a variadic template Variant, the type
    // completion_signatures<Fns...>::error_types<Variant> names the type
    // Variant<Error0, Error1, ... Errorn-1>, where m is the size of the
    // parameter pack ErrorFns.
    //
    // completion_signatures<Fns...>::sends_stopped is true if at least one of
    // the types in Fns is execution::set_stopped_t(); otherwise, false.
    //
    template <typename... Signatures>
    using completion_signatures = meta::type<
        detail::generate_completion_signatures<meta::pack<Signatures...>>>;

#if defined(HPX_HAVE_CXX20_COROUTINES)
    struct as_awaitable_t;
#endif

    namespace detail {

        struct completion_signals_of_sender_depend_on_execution_environment
        {
        };

#if defined(HPX_HAVE_CXX20_COROUTINES)
        // https://github.com/NVIDIA/stdexec/pull/733#issue-1537242117
        //
        // We were inconsistent about whether promise types were directly
        // queryable, or whether they implemented get_env. Now we expect
        // promise types to implement get_env, and if they don't they are
        // implicitly given an empty environment.
        //
        // In get_completion_signatures, we were testing for awaitability
        // using the Env arg as a promise type, which was picking up stray
        // await_transform functions that connect would not use,
        // leading to an inconsistency.
        //
        // -- Eric Niebler
        //
        // To be kept in sync with the promise type used in connect_awaitable
        template <typename Env>
        struct env_promise;
#endif

        template <typename Env>
        struct dependent_completion_signatures
        {
            template <template <typename...> typename,
                template <typename...> typename>
            using value_types =
                completion_signals_of_sender_depend_on_execution_environment;

            template <template <typename...> typename>
            using error_types =
                completion_signals_of_sender_depend_on_execution_environment;

            static constexpr bool sends_stopped = false;

#if defined(HPX_HAVE_CXX20_COROUTINES)
            bool await_ready() = delete;
            void await_suspend(
                hpx::coroutine_handle<env_promise<Env>>) = delete;
            dependent_completion_signatures await_resume() = delete;
#endif
        };

        template <>
        struct dependent_completion_signatures<no_env>
        {
#if defined(HPX_HAVE_CXX20_COROUTINES)
            bool await_ready();
            void await_suspend(hpx::coroutine_handle<env_promise<no_env>>);
            dependent_completion_signatures await_resume();
#endif
        };
    }    // namespace detail

    // dependent_completion_signatures is a placeholder completion signatures
    // descriptor that can be used to report that a type might be a sender
    // within a particular execution environment, but it isn't a sender in an
    // arbitrary execution environment.
    template <typename Env>
    using dependent_completion_signatures =
        detail::dependent_completion_signatures<Env>;

    namespace detail {

        template <typename Sender, typename Enable = void>
        struct has_completion_signatures : std::false_type
        {
        };

        template <typename Sender>
        struct has_completion_signatures<Sender,
            std::void_t<
                typename remove_cv_ref_t<Sender>::completion_signatures>>
          : std::true_type
        {
        };

        struct no_completion_signatures
        {
        };

        // sender<T> only checks if T is an awaitable if enable_sender<T> is
        // false. Then it checks for awaitability with a promise type that
        // doesn't have any environment queries, but that does have an
        // await_transform that pipes the T through
        // std::execution::as_awaitable. So you have two options for opting into
        // the sender concept if you type is not generally awaitable: (1)
        // specialize enable_sender, or (2) customize as_awaitable for T.
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(is_sender)

#ifdef HPX_HAVE_CXX20_COROUTINES
        template <typename Sender, typename = void>
        inline constexpr bool is_enable_sender_v = has_is_sender_v<Sender>;

        template <typename Sender>
        inline constexpr bool is_enable_sender_v<Sender,
            std::enable_if_t<
                std::is_move_constructible_v<std::decay_t<Sender>> &&
                std::is_class_v<std::decay_t<Sender>> &&
                hpx::util::is_detected_v<is_awaitable, Sender,
                    env_promise<no_env>>>> =
            is_awaitable_v<Sender, env_promise<no_env>> ||
            has_is_sender_v<Sender>;
#else
        template <typename Sender>
        inline constexpr bool is_enable_sender_v = has_is_sender_v<Sender>;
#endif
    }    // namespace detail

    // execution::get_completion_signatures is a customization point object. Let
    // s be an expression such that decltype((s)) is S, and let e be an
    // expression such that decltype((e)) is E. Then
    // get_completion_signatures(s) is expression-equivalent to
    //
    //      get_completion_signatures(s, no_env{})
    //
    //  and get_completion_signatures(s, e) is expression-equivalent to:
    //
    //      tag_invoke_result_t<get_completion_signatures_t, S, E>{} if that
    //      expression is well-formed,
    //
    // Otherwise, if remove_cvref_t<S>::completion_signatures is well-formed and
    // names a type, then a it returns a prvalue of
    // remove_cvref_t<S>::completion_signatures
    //
    // Otherwise, if is-awaitable<S> is true, then
    //
    // If await-result-type<S> is cv void then a prvalue of a type equivalent
    // to:
    //
    //      completion_signatures<
    //          set_value_t(),
    //          set_error_t(exception_ptr),
    //          set_stopped_t()>
    //
    //  Otherwise, a prvalue of a type equivalent to:
    //
    //      completion_signatures<
    //          set_value_t(await-result-type<S>),
    //          set_error_t(exception_ptr),
    //          set_stopped_t()>
    //
    //  Otherwise, no-completion-signatures{}.
    //
    inline constexpr struct get_completion_signatures_t final
      : hpx::functional::detail::tag_fallback<get_completion_signatures_t>
    {
    private:
        template <typename Sender, typename Env = no_env>
        friend constexpr auto tag_fallback_invoke(
            get_completion_signatures_t, Sender&&, Env const& = {}) noexcept
        {
            static_assert(sizeof(Sender),
                "Incomplete type used with get_completion_signatures");
            static_assert(sizeof(Env),
                "Incomplete type used with get_completion_signatures");

            if constexpr (meta::value<
                              detail::has_completion_signatures<Sender>>)
            {
                return typename detail::remove_cv_ref_t<
                    Sender>::completion_signatures{};
            }
#if defined(HPX_HAVE_CXX20_COROUTINES)
            else if constexpr (is_awaitable_v<Sender, detail::env_promise<Env>>)
            {
                using result_type =
                    await_result_t<Sender, detail::env_promise<Env>>;
                if constexpr (std::is_same_v<result_type,
                                  detail::dependent_completion_signatures<
                                      no_env>>)
                {
                    return detail::dependent_completion_signatures<no_env>{};
                }
                else
                {
                    return completion_signatures<
                        hpx::meta::invoke<
                            hpx::meta::remove<void,
                                hpx::meta::compose_func<set_value_t>>,
                            result_type>,
                        set_error_t(std::exception_ptr)>{};
                }
            }
#endif
            else if constexpr (std::is_same_v<Env, no_env> &&
                detail::is_enable_sender_v<std::decay_t<Sender>>)
            {
                return detail::dependent_completion_signatures<no_env>{};
            }
            else
            {
                return detail::no_completion_signatures{};
            }
        }
    } get_completion_signatures{};

    // A sender is a type that is describing an asynchronous operation. The
    // operation itself might not have started yet. In order to get the result
    // of this asynchronous operation, a sender needs to be connected to a
    // receiver with the corresponding value, error and stopped channels:
    //
    //     * `hpx::execution::experimental::connect`
    //
    // A sender describes a potentially asynchronous operation. A sender's
    // responsibility is to fulfill the receiver contract of a connected
    // receiver by delivering one of the receiver completion-signals.
    //
    // A sender's destructor shall not block pending completion of submitted
    // operations.
    template <typename Sender, typename Env = no_env>
    struct is_sender;

    // \see is_sender
    template <typename Sender, typename Receiver>
    struct is_sender_to;

    // The sender_of concept defines the requirements for a sender type that on
    // successful completion sends the specified set of value types.
    template <typename S, typename E = no_env, typename... Ts>
    struct is_sender_of;

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        constexpr bool has_value_types(
            typename T::template value_types<meta::pack, meta::pack>*) noexcept
        {
            return true;
        }

        template <typename T>
        constexpr bool has_value_types(...) noexcept
        {
            return false;
        }

        template <typename T>
        constexpr bool has_error_types(
            typename T::template error_types<meta::pack>*) noexcept
        {
            return true;
        }

        template <typename T>
        constexpr bool has_error_types(...) noexcept
        {
            return false;
        }

        template <typename T>
        constexpr bool has_sends_stopped(decltype(T::sends_stopped)*) noexcept
        {
            return true;
        }

        template <typename T>
        constexpr bool has_sends_stopped(...) noexcept
        {
            return false;
        }

        template <typename T>
        struct has_sender_types
          : std::integral_constant<bool,
                has_value_types<T>(nullptr) && has_error_types<T>(nullptr) &&
                    has_sends_stopped<T>(nullptr)>
        {
        };

        template <typename T>
        inline constexpr bool has_sender_types_v = has_sender_types<T>::value;

        ///////////////////////////////////////////////////////////////////////
        template <typename Traits>
        struct valid_completion_signatures
        {
            using type = Traits;
        };

        // leave undefined
        template <>
        struct valid_completion_signatures<no_completion_signatures>;

        template <typename Sender, typename Env = no_env>
        using completion_signatures_of = meta::type<valid_completion_signatures<
            util::invoke_result_t<get_completion_signatures_t, Sender, Env>>>;

        template <typename Sender, typename Env, bool IsClass,
            typename Enable = void>
        struct provides_completion_signatures_impl : std::false_type
        {
        };

        template <typename Sender, typename Env, typename Enable = void>
        struct completion_signatures_of_is_valid : std::false_type
        {
        };

        template <typename Sender, typename Env>
        struct completion_signatures_of_is_valid<Sender, Env,
            std::void_t<decltype(get_completion_signatures(
                std::declval<Sender>(), std::declval<Env>()))>> : std::true_type
        {
        };

        template <typename Sender, typename Env>
        struct provides_completion_signatures_impl<Sender, Env, true,
            std::enable_if_t<
                meta::value<completion_signatures_of_is_valid<Sender, Env>> &&
                has_sender_types_v<completion_signatures_of<Sender, Env>>>>
          : std::true_type
        {
        };

        template <typename Sender, typename Env = no_env>
        struct provides_completion_signatures
          : provides_completion_signatures_impl<Sender, Env,
                std::is_class_v<Sender>>
        {
        };
    }    // namespace detail

    // The alias template completion_signatures_of_t is used to query a sender
    // type for facts associated with the signals it sends.
    //
    // completion_signatures_of_t also recognizes awaitables as senders. For
    // this clause ([exec]):
    //
    // An awaitable is an expression that would be well-formed as the operand of
    // a co_await expression within a given context.
    //
    // For any type T, is-awaitable<T> is true if and only if an expression of
    // that type is an awaitable as described above within the context of a
    // coroutine whose promise type does not define a member await_transform.
    // For a coroutine promise type P, is-awaitable<T, P> is true if and only if
    // an expression of that type is an awaitable as described above within the
    // context of a coroutine whose promise type is P.
    //
    // For an awaitable a such that decltype((a)) is type A,
    // await-result-type<A> is an alias for decltype(e), where e is a's
    // await-resume expression ([expr.await]) within the context of a coroutine
    // whose promise type does not define a member await_transform. For a
    // coroutine promise type P, await-result-type<A, P> is an alias for
    // decltype(e), where e is a's await-resume expression ([expr.await]) within
    // the context of a coroutine whose promise type is P.
    //
    // For types S and E, the type completion_signatures_of_t<S, E> is an alias
    // for decltype(get_completion_signatures(declval<S>(), declval<E>())) if
    // that expression is well-formed and names a type other than
    // no-completion-signatures. Otherwise, it is ill-formed.
    //
    // The exposition-only type variant-or-empty<Ts...> is defined as follows:
    //
    // If sizeof...(Ts) is greater than zero, variant-or-empty<Ts...> names the
    // type variant<Us...> where Us... is the pack decay_t<Ts>... with duplicate
    // types removed.
    //
    // Otherwise, variant-or-empty<Ts...> names an implementation defined class
    // type equivalent to the following:
    //
    //      struct empty-variant {
    //          empty-variant() = delete;
    //      };
    //
    // Let r be an rvalue receiver of type R, and let S be the type of a sender.
    // If value_types_of_t<S, env_of_t<R>, Tuple, Variant> is well formed, it
    // shall name the type Variant<Tuple<Args0...>, Tuple<Args1...>, ...,
    // Tuple<ArgsN...>>>, where the type packs Args0 through ArgsN are the packs
    // of types the sender S passes as arguments to execution::set_value
    // (besides the receiver object). If such sender S odr-uses
    // ([basic.def.odr]) execution::set_value(r, args...), where
    // decltype(args)... is not one of the type packs Args0... through ArgsN...
    // (ignoring differences in rvalue-reference qualification), the program is
    // ill-formed with no diagnostic required.
    //
    // Let r be an rvalue receiver of type R, and let S be the type of a sender.
    // If error_types_of_t<S, env_of_t<R>, Variant> is well formed, it shall
    // name the type Variant<E0, E1, ..., EN>, where the types E0 through EN are
    // the types the sender S passes as arguments to execution::set_error
    // (besides the receiver object). If such sender S odr-uses
    // execution::set_error(r, e), where decltype(e) is not one of the types E0
    // through EN (ignoring differences in rvalue-reference qualification), the
    // program is ill-formed with no diagnostic required.
    //
    // Let r be an rvalue receiver of type R, and let S be the type of a sender.
    // If completion_signatures_of_t<S, env_of_t<R>>::sends_stopped is well
    // formed and false, and such sender S odr-uses execution::set_stopped(r),
    // the program is ill-formed with no diagnostic required.
    //
    // Let S be the type of a sender, let E be the type of an execution
    // environment other than execution::no_env such that sender<S, E> is true.
    // Let Tuple, Variant1, and Variant2 be variadic alias templates or class
    // templates such that following types are well-formed:
    //
    //      value_types_of_t<S, no_env, Tuple, Variant1>
    //
    //      error_types_of_t<S, no_env, Variant2>
    //
    // then the following shall also be true:
    //
    // value_types_of_t<S, E, Tuple, Variant1> shall also be well-formed and
    // shall name the same type as value_types_of_t<S, no_env, Tuple, Variant1>,
    //
    // error_types_of_t<S, E, Variant2> shall also be well-formed and shall name
    // the same type as error_types_of_t<S, no_env, Variant2>, and
    //
    // completion_signatures_of_t<S, E>::sends_stopped shall have the same value
    // as completion_signatures_of_t<S, no_env>::sends_stopped.
    template <typename Sender, typename Env = no_env>
    using completion_signatures_of_t =
        detail::completion_signatures_of<Sender, Env>;

    struct connect_t;

    namespace detail {

        template <bool IsSenderReceiver, typename Sender, typename Receiver>
        struct is_sender_to_impl;

        template <typename Sender, typename Receiver>
        struct is_sender_to_impl<false, Sender, Receiver> : std::false_type
        {
        };

        template <typename Sender, typename Receiver>
        struct is_sender_to_impl<true, Sender, Receiver>
          : std::integral_constant<bool,
                hpx::is_invocable_v<connect_t, Sender&&, Receiver&&> ||
                    hpx::is_invocable_v<connect_t, Sender&&, Receiver&> ||
                    hpx::is_invocable_v<connect_t, Sender&&, Receiver const&> ||
                    hpx::is_invocable_v<connect_t, Sender&, Receiver&&> ||
                    hpx::is_invocable_v<connect_t, Sender&, Receiver&> ||
                    hpx::is_invocable_v<connect_t, Sender&, Receiver const&> ||
                    hpx::is_invocable_v<connect_t, Sender const&, Receiver&&> ||
                    hpx::is_invocable_v<connect_t, Sender const&, Receiver&> ||
                    hpx::is_invocable_v<connect_t, Sender const&,
                        Receiver const&>>
        {
        };

        // This concept has been introduced to increase atomicity of concepts
        // clang-format off
        template <typename Sender, typename Env>
        inline constexpr bool is_sender_plain_v =
            std::is_move_constructible_v<std::decay_t<Sender>> &&
            std::is_class_v<std::decay_t<Sender>>;
        // clang-format on
    }    // namespace detail

    template <typename Sender, typename Env>
    struct is_sender
      : std::integral_constant<bool,
            !!(detail::is_sender_plain_v<Sender, Env> &&
                detail::is_enable_sender_v<std::decay_t<Sender>>) ||
                meta::value<detail::provides_completion_signatures<
                    std::decay_t<Sender>, Env>>>
    {
    };

    template <typename Sender, typename Env = no_env>
    inline constexpr bool is_sender_v = is_sender<Sender, Env>::value;

    namespace detail {

        template <bool IsSenderOf, typename S, typename E, typename... Ts>
        struct is_sender_of_impl;

        template <typename S, typename E, typename... Ts>
        struct is_sender_of_impl<false, S, E, Ts...> : std::false_type
        {
        };

        template <typename CS, typename... Ts>
        inline bool constexpr is_same_types = std::is_same_v<
            typename CS::template value_types<meta::pack, meta::pack>,
            meta::pack<meta::pack<Ts...>>>;

        template <class S, class E, class... Ts>
        struct is_sender_of_impl<true, S, E, Ts...>
          : std::integral_constant<bool,
                is_same_types<completion_signatures_of_t<S, E>, Ts...>>
        {
        };
    }    // namespace detail

    // The sender_of concept defines the requirements for a sender type that on
    // successful completion sends the specified set of value types.
    //
    //      template <typename S, typename E = no_env, typename... Ts>
    //      concept sender_of =
    //          sender<S, E> &&
    //          same_as<
    //              type-list<Ts...>,
    //              value_types_of_t<S, E, type-list, type_identity_t>
    //          >;
    //
    template <typename S, typename E, typename... Ts>
    struct is_sender_of
      : detail::is_sender_of_impl<is_sender_v<S, E>, S, E, Ts...>
    {
    };

    template <typename S, typename E = no_env, typename... Ts>
    inline constexpr bool is_sender_of_v = is_sender_of<S, E, Ts...>::value;

    template <typename Sender, typename Receiver>
    struct is_sender_to
      : detail::is_sender_to_impl<is_sender_v<Sender, env_of_t<Receiver>> &&
                is_receiver_v<Receiver>,
            Sender, Receiver>
    {
    };

    template <typename Sender, typename Receiver>
    inline constexpr bool is_sender_to_v =
        is_sender_to<Sender, Receiver>::value;

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        template <typename Signatures, typename Tuple, typename Variant>
        using value_types_from =
            typename Signatures::template value_types<Tuple::template apply,
                Variant::template apply>;

        template <typename Signatures, typename Variant>
        using error_types_from =
            typename Signatures::template error_types<Variant::template apply>;

        template <typename Signatures>
        using sends_stopped_from = meta::bool_<Signatures::sends_stopped>;
    }    // namespace detail

    // Let r be an rvalue receiver of type R, and let S be the type of a sender.
    // If value_types_of_t<S, env_of_t<R>, Tuple, Variant> is well formed, it
    // shall name the type Variant<Tuple<Args0...>, Tuple<Args1...>, ...,
    // Tuple<ArgsN...>>>, where the type packs Args0 through ArgsN are the packs
    // of types the sender S passes as arguments to execution::set_value
    // (besides the receiver object). If such sender S odr-uses
    // ([basic.def.odr]) execution::set_value(r, args...), where
    // decltype(args)... is not one of the type packs Args0... through ArgsN...
    // (ignoring differences in rvalue-reference qualification), the program is
    // ill-formed with no diagnostic required.
    //
    // clang-format off
    template <typename Sender, typename Env = no_env,
        template <typename...> typename Tuple = detail::decayed_tuple,
        template <typename...> typename Variant = detail::decayed_variant,
        typename = std::enable_if_t<is_sender_v<Sender, Env>>>
    // clang-format on
    using value_types_of_t =
        detail::value_types_from<detail::completion_signatures_of<Sender, Env>,
            meta::func<Tuple>, meta::func<Variant>>;

    /// Start definitions from coroutine_utils and sender

    template <typename Receiver, typename Sender>
    inline constexpr bool is_receiver_from_v = is_receiver_of_v<Receiver,
        completion_signatures_of_t<Sender, env_of_t<Receiver>>>;

    // Alias template single-sender-value-type is defined as follows:
    //
    // 1. If value_types_of_t<S, E, Tuple, Variant> would have the form
    // Variant<Tuple<T>>, then single-sender-value-type<S, E> is an alias for
    // type T.
    // 2. Otherwise, if value_types_of_t<S, E, Tuple, Variant> would
    // have the form Variant<Tuple<>> or Variant<>, then
    // single-sender-value-type<S, E> is an alias for type void.
    // 3. Otherwise, single-sender-value-type<S, E> is ill-formed.
    //
    template <typename Sender, typename Env = no_env>
    using single_sender_value_t =
        detail::value_types_from<detail::completion_signatures_of<Sender, Env>,
            meta::single_or<void>, meta::compose_template_func<meta::single_t>>;

    struct connect_awaitable_t;

    struct is_debug_env_t : hpx::functional::tag<is_debug_env_t>
    {
        template <typename Env,
            typename = std::enable_if_t<
                hpx::functional::is_tag_invocable_v<is_debug_env_t, Env>>>
        void tag_invoke(Env&&) const noexcept;
    };

    struct connect_t;

    template <typename S, typename R>
    using connect_result_t = hpx::util::invoke_result_t<connect_t, S, R>;

    template <typename Sender, typename Receiver>
    struct has_nothrow_connect;

#if defined(HPX_HAVE_CXX20_COROUTINES)
    // 4.18. Cancellation of a sender can unwind a stack of coroutines As
    // described in the section "All awaitables are senders", the sender
    // customization points recognize awaitables and adapt them transparently to
    // model the sender concept. When connect-ing an awaitable and a receiver,
    // the adaptation layer awaits the awaitable within a coroutine that
    // implements unhandled_stopped in its promise type. The effect of this is
    // that an "uncatchable" stopped exception propagates seamlessly out of
    // awaitables, causing execution::set_stopped to be called on the receiver.
    // Obviously, unhandled_stopped is a library extension of the coroutine
    // promise interface. Many promise types will not implement
    // unhandled_stopped. When an uncatchable stopped exception tries to
    // propagate through such a coroutine, it is treated as an unhandled
    // exception and terminate is called. The solution, as described above, is
    // to use a sender adaptor to handle the stopped exception before awaiting
    // it. It goes without saying that any future Standard Library coroutine
    // types ought to implement unhandled_stopped. The author of Add lazy
    // coroutine (coroutine task) type, which proposes a standard coroutine task
    // type, is in agreement.
    template <typename Promise, typename = void>
    inline constexpr bool has_unhandled_stopped = false;

    template <typename Promise>
    inline constexpr bool has_unhandled_stopped<Promise,
        std::void_t<decltype(std::declval<Promise>().unhandled_stopped())>> =
        true;

    template <typename Promise, typename = void>
    inline constexpr bool has_convertible_unhandled_stopped = false;

    template <typename Promise>
    inline constexpr bool has_convertible_unhandled_stopped<Promise,
        std::enable_if_t<std::is_convertible_v<
            decltype(std::declval<Promise>().unhandled_stopped()),
            hpx::coroutine_handle<>>>> = true;

    namespace detail {

        // clang-format off
        template <typename T, typename U>
        inline constexpr bool decays_to = std::is_same_v<std::decay_t<T>, U>&&
            std::is_same_v<std::decay_t<U>, T>;
        // clang-format on

        struct void_type
        {
        };

        template <typename Value>
        using value_or_void_t =
            hpx::meta::if_<std::is_same<Value, void>, void_type, Value>;

        template <typename Value>
        using coroutine_expected_result_t = hpx::variant<hpx::monostate,
            value_or_void_t<Value>, std::exception_ptr>;

        template <typename Promise>
        using coroutine_env_t = hpx::util::detected_or<exec_envs::empty_env,
            hpx::functional::tag_invoke_result_t, get_env_t, Promise>;

        template <typename Value>
        struct receiver_base
        {
            template <typename... Us,
                HPX_CONCEPT_REQUIRES_(
                    std::is_constructible_v<value_or_void_t<Value>, Us...>)>
            friend void tag_invoke(
                set_value_t, receiver_base&& self, Us&&... us) noexcept
            try
            {
                self.result->template emplace<1>(HPX_FORWARD(Us, us)...);
                self.continuation.resume();
            }
            catch (...)
            {
                set_error(
                    HPX_FORWARD(receiver_base, self), std::current_exception());
            }

            template <typename Error>
            friend void tag_invoke(
                set_error_t, receiver_base&& self, Error&& err) noexcept
            {
                if constexpr (decays_to<Error, std::exception_ptr>)
                {
                    self.result->template emplace<2>(HPX_FORWARD(Error, err));
                }
                else if constexpr (decays_to<Error, std::error_code>)
                {
                    self.result->template emplace<2>(
                        std::make_exception_ptr(std::system_error(err)));
                }
                else
                {
                    self.result->template emplace<2>(
                        std::make_exception_ptr(HPX_FORWARD(Error, err)));
                }
                self.continuation.resume();
            }

            coroutine_expected_result_t<Value>* result;
            hpx::coroutine_handle<> continuation;
        };

        template <typename PromiseId, typename Value>
        struct receiver
        {
            using Promise = hpx::meta::type<PromiseId>;
            struct type : receiver_base<Value>
            {
                using id = receiver;
                friend void tag_invoke(set_stopped_t, type&& self) noexcept
                {
                    auto continuation =
                        hpx::coroutine_handle<Promise>::from_address(
                            self.continuation.address());
                    hpx::coroutine_handle<> stopped_continuation =
                        continuation.promise().unhandled_stopped();
                    stopped_continuation.resume();
                }
                friend coroutine_env_t<Promise&> tag_invoke(
                    get_env_t, type const& self)
                {
                    if constexpr (hpx::functional::is_tag_invocable_v<get_env_t,
                                      Promise&>)
                    {
                        auto continuation =
                            hpx::coroutine_handle<Promise>::from_address(
                                self.continuation.address());
                        return get_env(continuation.promise());
                    }
                    else
                    {
                        return no_env{};
                    }
                }
            };
        };

        template <typename Sender, typename Promise>
        using receiver_t =
            hpx::meta::type<receiver<hpx::meta::get_id_t<Promise>,
                single_sender_value_t<Sender, coroutine_env_t<Promise>>>>;

        template <typename PromiseId, typename Value>
        struct sender_awaitable_base
        {
            static constexpr bool await_ready() noexcept
            {
                return false;
            }

            Value await_resume()
            {
                switch (result.index())
                {
                default:
                    [[fallthrough]];
                case 0:    // receiver contract not satisfied
                    HPX_ASSERT_MSG(0, "_Should never get here");
                    break;

                case 1:    // set_value
                    if constexpr (!std::is_void_v<Value>)
                    {
                        return HPX_MOVE(std::get<1>(result));
                    }
                    else
                    {
                        return;
                    }

                case 2:    // set_error
                    std::rethrow_exception(std::get<2>(result));
                }
                std::terminate();
            }

        protected:
            coroutine_expected_result_t<Value> result;
        };

        template <typename PromiseId, typename SenderId>
        struct sender_awaitable;

        template <typename Promise, typename Sender>
        using sender_awaitable_t =
            hpx::meta::type<sender_awaitable<hpx::meta::get_id_t<Promise>,
                hpx::meta::get_id_t<Sender>>>;

        // clang-format off
        template <typename Sender, typename Env = no_env>
        inline constexpr bool is_single_typed_sender_v =
            is_sender_v<Sender, Env> &&
            hpx::meta::value<
                hpx::meta::is_valid<single_sender_value_t, Sender, Env>>;

        template <typename Sender, typename Promise>
        inline constexpr bool is_awaitable_sender_v =
            is_single_typed_sender_v<Sender, coroutine_env_t<Promise>> &&
            is_sender_to_v<Sender, receiver_t<Sender, Promise>> &&
            has_unhandled_stopped<Promise>;
        // clang-format on
    }    // namespace detail

    inline constexpr struct as_awaitable_t
      : hpx::functional::detail::tag_fallback<as_awaitable_t>
    {
        template <typename T, typename Promise>
        static constexpr auto select_impl() noexcept
        {
            if constexpr (hpx::functional::is_tag_invocable_v<as_awaitable_t, T,
                              Promise&>)
            {
                using result_type =
                    hpx::functional::tag_invoke_result_t<as_awaitable_t, T,
                        Promise&>;
                constexpr bool Nothrow =
                    hpx::functional::is_nothrow_tag_invocable_v<as_awaitable_t,
                        T, Promise&>;
                return static_cast<result_type (*)() noexcept(Nothrow)>(
                    nullptr);
            }
            else if constexpr (is_awaitable_v<T>)
            {
                // NOT awaitable<T, Promise> !!
                using func_type = T && (*) () noexcept;
                return static_cast<func_type>(nullptr);
            }
            else if constexpr (detail::is_awaitable_sender_v<T, Promise>)
            {
                using result_type = detail::sender_awaitable_t<Promise, T>;
                constexpr bool Nothrow =
                    std::is_nothrow_constructible_v<result_type, T,
                        hpx::coroutine_handle<Promise>>;
                return static_cast<result_type (*)() noexcept(Nothrow)>(
                    nullptr);
            }
            else
            {
                using func_type = T && (*) () noexcept;
                return static_cast<func_type>(nullptr);
            }
        }

        template <typename T, typename Promise>
        using select_impl_t = decltype(select_impl<T, Promise>());

        template <typename T, typename Promise>
        friend HPX_FORCEINLINE auto
        tag_fallback_invoke(as_awaitable_t, T&& t, Promise& promise) noexcept(
            hpx::is_nothrow_invocable_v<select_impl_t<T, Promise>>)
            -> hpx::util::invoke_result_t<select_impl_t<T, Promise>>
        {
            if constexpr (detail::is_awaitable_sender_v<T, Promise>)
            {
                auto hcoro =
                    hpx::coroutine_handle<Promise>::from_promise(promise);
                return detail::sender_awaitable_t<Promise, T>{
                    HPX_FORWARD(T, t), hcoro};
            }
            else
            {
                return HPX_FORWARD(T, t);
            }
        }
    } as_awaitable{};

    namespace detail {

        template <typename Env>
        struct env_promise
        {
            template <typename Ty,
                typename =
                    std::enable_if_t<!hpx::functional::is_tag_invocable_v<
                        as_awaitable_t, Ty, env_promise&>>>
            Ty&& await_transform(Ty&& value) noexcept
            {
                return HPX_FORWARD(Ty, value);
            }

            template <typename Ty,
                typename = std::enable_if_t<hpx::functional::is_tag_invocable_v<
                    as_awaitable_t, Ty, env_promise&>>>
            auto await_transform(Ty&& value) noexcept(
                hpx::functional::is_nothrow_tag_invocable_v<as_awaitable_t, Ty,
                    env_promise&>)
                -> hpx::functional::tag_invoke_result_t<as_awaitable_t, Ty,
                    env_promise&>
            {
                return tag_invoke(as_awaitable, HPX_FORWARD(Ty, value), *this);
            }

            template <typename T>
            friend auto tag_invoke(get_env_t, env_promise<T> const&) noexcept
                -> T const&;
        };

        struct with_awaitable_senders_base
        {
            with_awaitable_senders_base() = default;

            template <typename OtherPromise>
            void set_continuation(
                hpx::coroutine_handle<OtherPromise> hcoro) noexcept
            {
                static_assert(!std::is_void_v<OtherPromise>);

                continuation_handle = hcoro;
                if constexpr (has_unhandled_stopped<OtherPromise>)
                {
                    stopped_callback =
                        [](void* address) noexcept -> hpx::coroutine_handle<> {
                        // This causes the rest of the coroutine (the part after
                        // the co_await of the sender) to be skipped and invokes
                        // the calling coroutine's stopped handler.
                        return hpx::coroutine_handle<
                            OtherPromise>::from_address(address)
                            .promise()
                            .unhandled_stopped();
                    };
                }

                // If OtherPromise doesn't implement unhandled_stopped(),
                // then if a "stopped" unwind reaches this point,
                // it's considered an unhandled exception and terminate()
                // is called.
            }

            hpx::coroutine_handle<> continuation() const noexcept
            {
                return continuation_handle;
            }

            hpx::coroutine_handle<> unhandled_stopped() const noexcept
            {
                return (*stopped_callback)(continuation_handle.address());
            }

        private:
            hpx::coroutine_handle<> continuation_handle{};
            hpx::coroutine_handle<> (*stopped_callback)(void*) noexcept =
                [](void*) noexcept -> hpx::coroutine_handle<> {
                std::terminate();
            };
        };
    }    // namespace detail

    // clang-format off
    template <typename A, typename B>
    inline constexpr bool is_derived_from_v = std::is_base_of_v<B, A> &&
        std::is_convertible_v<A const volatile*, B const volatile*>;
    // clang-format on

    template <typename Promise>
    struct with_awaitable_senders : detail::with_awaitable_senders_base
    {
        template <typename Value>
        auto await_transform(Value&& val)
            -> hpx::util::invoke_result_t<as_awaitable_t, Value, Promise&>
        {
            static_assert(is_derived_from_v<Promise, with_awaitable_senders>);
            return as_awaitable(
                HPX_FORWARD(Value, val), static_cast<Promise&>(*this));
        }
    };

    struct promise_base
    {
        static constexpr hpx::suspend_always initial_suspend() noexcept
        {
            return {};
        }

        [[noreturn]] static hpx::suspend_always final_suspend() noexcept
        {
            std::terminate();
        }

        [[noreturn]] static void unhandled_exception() noexcept
        {
            std::terminate();
        }

        [[noreturn]] static void return_void() noexcept
        {
            std::terminate();
        }

        template <typename Fun>
        auto yield_value(Fun&& fun) noexcept
        {
            struct awaiter
            {
                Fun&& fun;

                static constexpr bool await_ready() noexcept
                {
                    return false;
                }

                void await_suspend(hpx::coroutine_handle<>) noexcept(
                    std::is_nothrow_invocable_v<Fun>)
                {
                    // If this throws, the runtime catches the exception,
                    // resumes the connect_awaitable coroutine, and immediately
                    // rethrows the exception. The end result is that an
                    // exception_ptr to the exception gets passed to set_error.
                    HPX_FORWARD(Fun, fun)();
                }

                [[noreturn]] static void await_resume() noexcept
                {
                    std::terminate();
                }
            };

            return awaiter{HPX_FORWARD(Fun, fun)};
        }
    };

    struct operation_base
    {
        hpx::coroutine_handle<> coro_handle;

        explicit operation_base(hpx::coroutine_handle<> hcoro) noexcept
          : coro_handle(hcoro)
        {
        }

        operation_base(operation_base const& other) = delete;
        operation_base(operation_base&& other) noexcept
          : coro_handle(std::exchange(other.coro_handle, {}))
        {
        }

        operation_base& operator=(operation_base const&) = delete;
        operation_base& operator=(operation_base&& rhs) noexcept
        {
            coro_handle = std::exchange(rhs.coro_handle, {});
            return *this;
        }

        ~operation_base()
        {
            if (coro_handle)
                coro_handle.destroy();
        }

        friend void tag_invoke(start_t, operation_base& self) noexcept
        {
            self.coro_handle.resume();
        }
    };

    template <typename ReceiverId>
    struct promise;

    template <typename ReceiverId>
    struct operation
    {
        struct type : operation_base
        {
            using promise_type = hpx::meta::type<promise<ReceiverId>>;
            using operation_base::operation_base;
        };
    };

    template <typename ReceiverId>
    struct promise
    {
        using Receiver = hpx::meta::type<ReceiverId>;
        struct type : promise_base
        {
            using id = promise;
            explicit type(auto&, Receiver& rcvr_) noexcept
              : rcvr(rcvr_)
            {
            }

            hpx::coroutine_handle<> unhandled_stopped() noexcept
            {
                set_stopped(std::move(rcvr));
                // Returning noop_coroutine here causes the __connect_awaitable
                // coroutine to never resume past the point where it co_await's
                // the awaitable.
                return hpx::noop_coroutine();
            }

            hpx::meta::type<operation<ReceiverId>> get_return_object() noexcept
            {
                return hpx::meta::type<operation<ReceiverId>>{
                    hpx::coroutine_handle<type>::from_promise(*this)};
            }

            template <typename Awaitable>
            auto await_transform(Awaitable&& await) noexcept(
                hpx::functional::is_nothrow_tag_invocable_v<as_awaitable_t,
                    Awaitable, promise&> ||
                !hpx::functional::is_tag_invocable_v<as_awaitable_t, Awaitable,
                    promise&>)

            {
                if constexpr (hpx::functional::is_tag_invocable_v<
                                  as_awaitable_t, Awaitable, promise&>)
                {
                    return as_awaitable(HPX_FORWARD(Awaitable, await), *this);
                }
                else
                {
                    return HPX_FORWARD(Awaitable, await);
                }
            }

            // Pass through the get_env receiver query
            friend auto tag_invoke(get_env_t, type const& self)
                -> env_of_t<Receiver>
            {
                return get_env(self.rcvr);
            }

            Receiver& rcvr;
        };
    };

    template <typename Receiver,
        typename = std::enable_if_t<is_receiver_v<Receiver>>>
    using promise_t = hpx::meta::type<promise<hpx::meta::get_id_t<Receiver>>>;

    template <typename Receiver,
        typename = std::enable_if_t<is_receiver_v<Receiver>>>
    using operation_t =
        hpx::meta::type<operation<hpx::meta::get_id_t<Receiver>>>;

    inline constexpr struct connect_awaitable_t
    {
    private:
        template <typename Fun, typename... Ts>
        static auto co_call(Fun fun, Ts&&... as) noexcept
        {
            auto fn = [&, fun]() noexcept { fun(HPX_FORWARD(Ts, as)...); };

            struct awaiter
            {
                decltype(fn) fn_;

                static constexpr bool await_ready() noexcept
                {
                    return false;
                }

                void await_suspend(hpx::coroutine_handle<>) noexcept
                {
                    fn_();
                }

                [[noreturn]] static void await_resume() noexcept
                {
                    std::terminate();
                }
            };

            return awaiter{fn};
        }

        template <typename Awaitable, typename Receiver>
        static operation_t<Receiver> impl(Awaitable await, Receiver rcvr)
        {
            using result_t = await_result_t<Awaitable, promise_t<Receiver>>;
            std::exception_ptr eptr;
            try
            {
                if constexpr (std::is_void_v<result_t>)
                {
                    // clang-format off
                    co_await (co_await HPX_FORWARD(Awaitable, await),
                        co_call(set_value, HPX_FORWARD(Receiver, rcvr)));
                    // clang-format on
                }
                else
                {
                    co_await co_call(set_value, HPX_FORWARD(Receiver, rcvr),
                        co_await HPX_FORWARD(Awaitable, await));
                }
            }
            catch (...)
            {
                eptr = std::current_exception();
            }

            if (eptr)
            {
                co_await co_call(set_error, HPX_FORWARD(Receiver, rcvr),
                    HPX_FORWARD(std::exception_ptr, eptr));
            }
        }

        template <typename Receiver, typename Awaitable,
            typename = std::enable_if_t<is_receiver_v<Receiver>>>
        using completions_t = completion_signatures<
            hpx::meta::invoke1<    // set_value_t() or set_value_t(T)
                hpx::meta::remove<void, hpx::meta::compose_func<set_value_t>>,
                await_result_t<Awaitable, promise_t<Receiver>>>,
            set_error_t(std::exception_ptr), set_stopped_t()>;

    public:
        template <typename Receiver, typename Awaitable,
            typename = std::enable_if_t<
                is_awaitable_v<Awaitable, promise_t<Receiver>>>,
            typename = std::enable_if_t<
                is_receiver_of_v<Receiver, completions_t<Receiver, Awaitable>>>>
        operation_t<Receiver> operator()(Awaitable&& await, Receiver rcvr) const
        {
            return impl(
                HPX_FORWARD(Awaitable, await), HPX_FORWARD(Receiver, rcvr));
        }
    } connect_awaitable{};
#endif    // HPX_HAVE_CXX20_COROUTINES

    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct connect_t
      : hpx::functional::detail::tag_fallback<connect_t>
    {
#if defined(HPX_HAVE_CXX20_COROUTINES)
        template <typename Sender, typename Receiver,
            typename = std::enable_if_t<
                hpx::is_invocable_v<connect_awaitable_t, Sender, Receiver> ||
                hpx::functional::is_tag_invocable_v<is_debug_env_t,
                    env_of_t<Receiver>>>>
        friend constexpr auto tag_fallback_invoke(
            connect_t, Sender&& sndr, Receiver&& rcvr) noexcept
        {
            if constexpr (hpx::is_invocable_v<connect_awaitable_t, Sender,
                              Receiver>)
            {
                return connect_awaitable(
                    HPX_FORWARD(Sender, sndr), HPX_FORWARD(Receiver, rcvr));
            }
            else
            {
                return;
            }
        }
#endif    // HPX_HAVE_CXX20_COROUTINES
    } connect{};

    template <typename Sender, typename Receiver>
    struct has_nothrow_connect
      : std::integral_constant<bool,
            noexcept(connect(std::declval<Sender>(), std::declval<Receiver>()))>
    {
    };

#if defined(HPX_HAVE_CXX20_COROUTINES)
    namespace detail {

        template <typename PromiseId, typename SenderId>
        struct sender_awaitable
        {
            using promise_type = hpx::meta::type<PromiseId>;
            using sender_type = hpx::meta::type<SenderId>;
            using env_type = env_of_t<promise_type>;
            using value_type = single_sender_value_t<sender_type, env_type>;

            struct type : sender_awaitable_base<PromiseId, value_type>
            {
                // clang-format off
                type(sender_type&& sender, hpx::coroutine_handle<promise_type> hcoro)
                    noexcept(has_nothrow_connect<sender_type, receiver>::value)
                  : op_state_(connect(HPX_FORWARD(sender_type, sender),
                        receiver{{&this->result, hcoro}}))
                {
                }
                // clang-format on

                void await_suspend(hpx::coroutine_handle<promise_type>) noexcept
                {
                    start(op_state_);
                }

            private:
                using receiver = receiver_t<sender_type, promise_type>;
                connect_result_t<sender_type, receiver> op_state_;
            };
        };
    }     // namespace detail
#endif    // HPX_HAVE_CXX20_COROUTINES

    /// End definitions from coroutine_utils and sender

    // Let r be an rvalue receiver of type R, and let S be the type of a sender.
    // If error_types_of_t<S, env_of_t<R>, Variant> is well formed, it shall
    // name the type Variant<E0, E1, ..., EN>, where the types E0 through EN are
    // the types the sender S passes as arguments to execution::set_error
    // (besides the receiver object). If such sender S odr-uses
    // execution::set_error(r, e), where decltype(e) is not one of the types E0
    // through EN (ignoring differences in rvalue-reference qualification), the
    // program is ill-formed with no diagnostic required.
    //
    // clang-format off
    template <typename Sender, typename Env = no_env,
        template <typename...> typename Variant = detail::decayed_variant,
        typename = std::enable_if_t<is_sender_v<Sender, Env>>>
    // clang-format on
    using error_types_of_t =
        detail::error_types_from<detail::completion_signatures_of<Sender, Env>,
            meta::func<Variant>>;

    // clang-format off
    template <typename Sender, typename Env = no_env,
        typename = std::enable_if_t<is_sender_v<Sender, Env>>>
    // clang-format on
    inline constexpr bool sends_stopped_of_v =
        meta::value<detail::sends_stopped_from<
            detail::completion_signatures_of<Sender, Env>>>;

    namespace detail {

        // meta function versions for value_types_of, error_types_of, and
        // sends_stopped
        //
        // Note: the value_types_of and error_types_of functions below do not
        //       create an 'empty-variant' (this is different from the
        //       corresponding functions execution::value_types_of_t and
        //       execution::error_types_of_t).

        // clang-format off
        template <typename Sender, typename Env,
            typename Tuple  = meta::func<meta::pack>,
            typename Variant = meta::func<meta::pack>,
            typename = std::enable_if_t<is_sender_v<Sender, Env>>>
        // clang-format on
        using value_types_of =
            value_types_from<completion_signatures_of_t<Sender, Env>,
                decay_tuple<Tuple::template apply>,
                decay_variant<Variant::template apply>>;

        // clang-format off
        template <typename Sender, typename Env,
            typename Variant = meta::func<meta::pack>,
            typename = std::enable_if_t<is_sender_v<Sender, Env>>>
        // clang-format on
        using error_types_of =
            error_types_from<completion_signatures_of_t<Sender, Env>,
                decay_variant<Variant::template apply>>;

        // clang-format off
        template <typename Sender, typename Env = no_env,
            typename = std::enable_if_t<is_sender_v<Sender, Env>>>
        // clang-format on
        using sends_stopped_of =
            meta::bool_<completion_signatures_of_t<Sender, Env>::sends_stopped>;

        // helpers for make_completion_signatures
        template <typename... Ts>
        using set_value_signature = set_value_t(Ts...);

        template <typename Error>
        using set_error_signature = set_error_t(Error);

        template <typename Sender, typename Env, typename Signatures,
            typename SetValue, typename SetError, typename SendsStopped>
        using completion_signatures_t = meta::apply<
            meta::remove<void, meta::unique<meta::func<completion_signatures>>>,
            typename Signatures::signatures_t,
            value_types_of<Sender, Env, SetValue, meta::func<meta::pack>>,
            error_types_of<Sender, Env,
                meta::transform<SetError, meta::func<meta::pack>>>,
            meta::if_<SendsStopped, meta::pack<set_stopped_t()>, meta::pack<>>>;

        template <typename Sender, typename Env, typename Signatures,
            typename SetValue, typename SetError, bool SendsStopped>
        struct make_helper
        {
            using type = completion_signatures_t<Sender, Env, Signatures,
                SetValue, SetError, meta::bool_<SendsStopped>>;
        };

        template <typename Sender, typename Signatures, typename SetValue,
            typename SetError, bool SendsStopped>
        struct make_helper<Sender, no_env, Signatures, SetValue, SetError,
            SendsStopped>
        {
            using type = dependent_completion_signatures<no_env>;
        };
    }    // namespace detail

    // make_completion_signatures is an alias template used to adapt the
    // completion signatures of a sender. It takes a sender, and environment,
    // and several other template arguments that apply modifications to the
    // sender's completion signatures to generate a new instantiation of
    // execution::completion_signatures.
    //
    // Example:
    //
    // Given a sender S and an environment Env, adapt a S's completion
    // signatures by lvalue-ref qualifying the values, adding an additional
    // exception_ptr error completion if its not already there, and leaving the
    // other signals alone.
    //
    //      template <typename... Args> using my_set_value_t =
    //          execution::set_value_t(add_lvalue_reference_t<Args>...);
    //
    //      using my_completion_signals =
    //          execution::make_completion_signatures<
    //              S, Env, execution::completion_signatures<
    //                  execution::set_error_t(exception_ptr)>,
    //              my_set_value_t>;
    //
    // AddlSigs shall name an instantiation of the
    // execution::completion_signatures class template.
    //
    // SetValue shall name an alias template such that for any template
    // parameter pack As..., SetValue<As...> is either ill-formed, void or an
    // alias for a function type whose return type is execution::set_value_t.
    //
    // SetError shall name an alias template such that for any type Err,
    // SetError<Err> is either ill-formed, void or an alias for a function type
    // whose return type is execution::set_error_t.
    //
    // Let Vs... be a pack of the non-void types in the type-list named by
    // value_types_of_t<Sndr, Env, SetValue, type-list>.
    //
    // Let Es... be a pack of the non-void types in the type-list named by
    // error_types_of_t<Sndr, Env, error-list>, where error-list is an alias
    // template such that error-list<Ts...> names type-list<SetError<Ts>...>.
    //
    // Let Ss... be an empty pack if SendsStopped is false; otherwise, a pack
    // containing the single type execution::set_stopped_t().
    //
    // Let MoreSigs... be a pack of the template arguments of the
    // execution::completion_signatures instantiation named by AddlSigs.
    //
    // If any of the above types are ill-formed, then
    // make_completion_signatures<Sndr, Env, AddlSigs, SetValue, SetDone,
    // SendsStopped> is an alias for dependent_completion_signatures<Env>.
    //
    // Otherwise, make_completion_signatures<Sndr, Env, AddlSigs, SetValue,
    // SetDone, SendsStopped> names the type completion_signatures<Sigs...>
    // where Sigs... is the unique set of types in [Vs..., Es..., Ss...,
    // MoreSigs...].
    //
    // clang-format off
    template <typename Sender, typename Env = no_env,
        typename AddlSignatures = completion_signatures<>,
        template <typename...> typename SetValue = detail::set_value_signature,
        template <typename> typename SetError = detail::set_error_signature,
        bool SendsStopped =
            completion_signatures_of_t<Sender, Env>::sends_stopped,
        typename = std::enable_if_t<is_sender_v<Sender, Env>>>
    // clang-format on
    using make_completion_signatures =
        meta::type<detail::make_helper<Sender, Env, AddlSignatures,
            meta::func<SetValue>, meta::func1<SetError>, SendsStopped>>;
}    // namespace hpx::execution::experimental
