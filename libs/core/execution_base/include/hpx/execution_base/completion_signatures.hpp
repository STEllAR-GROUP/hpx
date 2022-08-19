//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/datastructures/variant.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/execution_base/traits/coroutine_traits.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/type_support/meta.hpp>
#include <hpx/type_support/pack.hpp>

#include <exception>
#include <type_traits>

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

        template <typename Signature, typename Tag,
            typename MetaF = meta::func<meta::pack>>
        using signature_arg_apply =
            decltype(test_signature<Tag, MetaF>((Signature*) nullptr));

        template <typename Signature, typename Enable = void>
        struct is_completion_signature : std::false_type
        {
        };

        template <typename Signature>
        struct is_completion_signature<Signature,
            std::void_t<decltype(test_signature((Signature*) nullptr))>>
          : std::true_type
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
    // Variant<Tuple<Values0...>, Tuple<Values1...>, ... Tuple<Valuesm-1...>>,
    // where m is the size of the parameter pack ValueFns.
    //
    // Let ErrorFns be a template parameter pack of the function types in Fns
    // whose return types are execution::set_error_t, and let Errorn be the
    // function argument type in the n-th type in ErrorFns.
    // Then, given a variadic template Variant, the type
    // completion_signatures<Fns...>::error_types<Variant> names the type
    // Variant<Error0, Error1, ... Errorm-1>, where m is the size of the
    // parameter pack ErrorFns.
    //
    // completion_signatures<Fns...>::sends_stopped is true if at least one of
    // the types in Fns is execution::set_stopped_t(); otherwise, false.
    //
    template <typename... Signatures>
    using completion_signatures = meta::type<
        detail::generate_completion_signatures<meta::pack<Signatures...>>>;

    namespace detail {

        struct completion_signals_of_sender_depend_on_execution_environment
        {
        };

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
        };

        template <>
        struct dependent_completion_signatures<no_env>
        {
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
            // TODO: Handle a case where is_awaitable_v<Sender,Promise>
            // where Promise type is not void.
            else if constexpr (is_awaitable_v<Sender>)
            {
                using result_type = await_result_t<Sender>;
                if constexpr (std::is_void_v<result_type>)
                {
                    return completion_signatures<set_value_t(),
                        set_error_t(std::exception_ptr)>{};
                }
                else
                {
                    return completion_signatures<set_value_t(result_type),
                        set_error_t(std::exception_ptr)>{};
                }
            }
#endif
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
            typename T::template value_types<meta::pack, meta::pack>*)
        {
            return true;
        }

        template <typename T>
        constexpr bool has_value_types(...)
        {
            return false;
        }

        template <typename T>
        constexpr bool has_error_types(
            typename T::template error_types<meta::pack>*)
        {
            return true;
        }

        template <typename T>
        constexpr bool has_error_types(...)
        {
            return false;
        }

        template <typename T>
        constexpr bool has_sends_stopped(decltype(T::sends_stopped)*)
        {
            return true;
        }

        template <typename T>
        constexpr bool has_sends_stopped(...)
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

        template <typename Sender, typename Env>
        struct provides_completion_signatures_impl<Sender, Env, true,
            std::enable_if_t<meta::value<meta::is_valid<
                                 completion_signatures_of, Sender, Env>> &&
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
    }    // namespace detail

    template <typename Sender, typename Env>
    struct is_sender
      : std::integral_constant<bool,
            std::is_move_constructible_v<std::decay_t<Sender>> &&
                std::is_class_v<std::decay_t<Sender>> &&
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

    template <typename Sender, typename Env = no_env>
    using single_sender_value_t = value_types_of_t<Sender, Env>;

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
    //      template <typename... Args>
    //      using my_set_value_t =
    //          execution::set_value_t(add_lvalue_reference_t<Args>...);
    //
    //      using my_completion_signals =
    //          execution::make_completion_signatures<
    //              S, Env,
    //              execution::completion_signatures<
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
