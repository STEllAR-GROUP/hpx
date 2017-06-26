//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_UNWRAP_IMPL_HPP
#define HPX_UTIL_DETAIL_UNWRAP_IMPL_HPP

#include <hpx/config.hpp>
#include <hpx/traits/future_traits.hpp>
#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_tuple_like.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/pack_traversal.hpp>
#include <hpx/util/result_of.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx {
namespace util {
    namespace detail {
        /// This struct defines a configurateable environment in order
        /// to adapt the unwrapping behaviour.
        ///
        /// The old unwrap (unwrapped) allowed futures to be instanted
        /// with void whereas the new one doesn't.
        ///
        /// The old unwrap (unwrapped) also unwraps the first occuring future
        /// independently a single value or multiple ones were passed to it.
        /// This behaviour is inconsistent and was removed in the
        /// new implementation.
        ///
        /// \note If the old implementation of unwrapped gets removed all
        ///       the configuration mechanis here can be removed while
        ///       preserving the behaviour of the new implementations
        ///       configuration.
        ///
        template <bool AllowsVoidFutures, bool UnwrapTopLevelTuples>
        struct unwrap_config
        {
            static HPX_CONSTEXPR_OR_CONST bool allows_void_futures =
                AllowsVoidFutures;

            static HPX_CONSTEXPR_OR_CONST bool unwrap_top_level_tuples =
                UnwrapTopLevelTuples;
        };

        /// The old unwrapped implementation
        using old_unwrap_config = unwrap_config<true, true>;
        /// The new unwrap implementation
        using new_unwrap_config = unwrap_config<false, false>;

        /// A tag which may replace void results when unwrapping
        struct unwrapped_void_tag
        {
        };

        /// Deduces to a true_type if the given future is instantiated with
        /// a non void type.
        template <typename T>
        using is_non_void_future = std::integral_constant<bool,
            traits::is_future<T>::value &&
                !std::is_void<
                    typename traits::future_traits<T>::result_type>::value>;

        /// Deduces to a true_type if the given future is instantiated with void
        template <typename T>
        using is_void_future = std::integral_constant<bool,
            traits::is_future<T>::value &&
                std::is_void<
                    typename traits::future_traits<T>::result_type>::value>;

        /// Return an unknown type (unwrapped_void_tag) when unwrapping
        /// futures instantiated with void to work around the issue that
        /// we can't represent void otherwise in a variadic argument pack.
        template <typename Config, typename T>
        unwrapped_void_tag void_or_assert(T&& future)
        {
            static_assert(
                Config::allows_void_futures && std::is_same<T, T>::value,
                "Unwrapping future<void> or shared_future<void> is "
                "forbidden! Use hpx::lcos::wait_all instead!");
            std::forward<T>(future).get();
            return {};
        }

        /// A mapper that maps futures to its representing type
        ///
        /// The mapper does unwrap futures nested inside futures until
        /// the particular given depth.
        ///
        /// - Depth >  1 -> Depth remaining
        /// - Depth == 1 -> One depth remaining
        /// - Depth == 0 -> Unlimited depths
        template <std::size_t Depth, typename Config>
        struct future_unwrap_until_depth
        {
            /// This piece of code can't be refactored out using
            /// inheritance and `using Base::operator()` because this
            /// isn't taken into account when doing SFINAE.
            template <typename T,
                typename std::enable_if<is_void_future<
                    typename std::decay<T>::type>::value>::type* = nullptr>
            auto operator()(T&& future) const
                -> decltype(void_or_assert<Config>(std::forward<T>(future)))
            {
                return void_or_assert<Config>(std::forward<T>(future));
            }

            template <typename T,
                typename std::enable_if<is_non_void_future<
                    typename std::decay<T>::type>::value>::type* = nullptr>
            auto operator()(T&& future) const -> decltype(
                map_pack(future_unwrap_until_depth<Depth - 1, Config>{},
                    std::forward<T>(future).get()))
            {
                return map_pack(future_unwrap_until_depth<Depth - 1, Config>{},
                    std::forward<T>(future).get());
            }
        };
        template <typename Config>
        struct future_unwrap_until_depth<1U, Config>
        {
            /// This piece of code can't be refactored out using
            /// inheritance and `using Base::operator()` because this
            /// isn't taken into account when doing SFINAE.
            template <typename T,
                typename std::enable_if<is_void_future<
                    typename std::decay<T>::type>::value>::type* = nullptr>
            auto operator()(T&& future) const
                -> decltype(void_or_assert<Config>(std::forward<T>(future)))
            {
                return void_or_assert<Config>(std::forward<T>(future));
            }

            template <typename T,
                typename std::enable_if<is_non_void_future<
                    typename std::decay<T>::type>::value>::type* = nullptr>
            auto operator()(T&& future) const -> typename traits::future_traits<
                typename std::decay<T>::type>::result_type
            {
                return std::forward<T>(future).get();
            }
        };
        template <typename Config>
        struct future_unwrap_until_depth<0U, Config>
        {
            /// This piece of code can't be refactored out using
            /// inheritance and `using Base::operator()` because this
            /// isn't taken into account when doing SFINAE.
            template <typename T,
                typename std::enable_if<is_void_future<
                    typename std::decay<T>::type>::value>::type* = nullptr>
            auto operator()(T&& future) const
                -> decltype(void_or_assert<Config>(std::forward<T>(future)))
            {
                return void_or_assert<Config>(std::forward<T>(future));
            }

            template <typename T,
                typename std::enable_if<is_non_void_future<
                    typename std::decay<T>::type>::value>::type* = nullptr>
            auto operator()(T&& future) const -> decltype(
                map_pack(std::declval<future_unwrap_until_depth const&>(),
                    std::forward<T>(future).get()))
            {
                return map_pack(*this, std::forward<T>(future).get());
            }
        };

        /// Unwraps the futures contained in the given pack args
        /// until the depth Depth.
        template <std::size_t Depth, typename Config, typename... Args>
        auto unwrap_depth_impl(Config, Args&&... args)
            -> decltype(map_pack(future_unwrap_until_depth<Depth, Config>{},
                std::forward<Args>(args)...))
        {
            return map_pack(future_unwrap_until_depth<Depth, Config>{},
                std::forward<Args>(args)...);
        }

        /// We use a specialized class here because MSVC has issues with
        /// tag dispatching a function because it does semantical checks before
        /// matching the tag, which leads to false errors.
        template <bool IsFusedInvoke>
        struct invoke_wrapped_impl
        {
            /// Invoke the callable with the tuple argument through invoke_fused
            template <typename C, typename T>
            static auto apply(C&& callable, T&& unwrapped)
                // There is no trait for the invoke_fused result
                -> decltype(invoke_fused(std::forward<C>(callable),
                    std::forward<T>(unwrapped)))
            {
                return invoke_fused(
                    std::forward<C>(callable), std::forward<T>(unwrapped));
            }
        };
        template <>
        struct invoke_wrapped_impl<false /*IsFusedInvoke*/>
        {
            /// Invoke the callable with the plain argument through invoke,
            /// also when the result is a tuple like type, when we received
            /// a single argument.
            template <typename C, typename T>
            static auto apply(C&& callable, T&& unwrapped) ->
                typename invoke_result<C, T>::type
            {
                return util::invoke(
                    std::forward<C>(callable), std::forward<T>(unwrapped));
            }
        };

        /// Indicates whether we should invoke the result through invoke_fused:
        /// - We called the function with multiple arguments.
        /// - The result is a tuple like type and UnwrapTopLevelTuples is set.
        template <bool HadMultipleArguments, bool UnwrapTopLevelTuples,
            typename Result>
        using should_fuse_invoke = std::integral_constant<bool,
            HadMultipleArguments ||
                (UnwrapTopLevelTuples &&
                    traits::is_tuple_like<
                        typename std::decay<Result>::type>::value)>;

        /// map_pack may return a tuple or a plain type, choose the
        /// corresponding invocation function accordingly.
        template <bool HadMultipleArguments, typename Config, typename C,
            typename T>
        auto invoke_wrapped(Config, C&& callable, T&& unwrapped) -> decltype(
            invoke_wrapped_impl<should_fuse_invoke<HadMultipleArguments,
                Config::unwrap_top_level_tuples, T>::value>::
                apply(std::forward<C>(callable), std::forward<T>(unwrapped)))
        {
            return invoke_wrapped_impl<should_fuse_invoke<HadMultipleArguments,
                Config::unwrap_top_level_tuples, T>::value>::
                apply(std::forward<C>(callable), std::forward<T>(unwrapped));
        }

        /// Implements the callable object which is returned by n invocation
        /// to hpx::util::unwrap and similar functions.
        template <typename Config, typename T, std::size_t Depth>
        class functional_unwrap_impl
        {
            /// The wrapped callable object
            T wrapped_;

        public:
            explicit functional_unwrap_impl(T wrapped)
              : wrapped_(std::move(wrapped))
            {
            }

            template <typename... Args>
            auto operator()(Args&&... args)
                -> decltype(invoke_wrapped<(sizeof...(args) > 1)>(Config{},
                    std::declval<T&>(),
                    unwrap_depth_impl<Depth>(Config{},
                        std::forward<Args>(args)...)))
            {
                return invoke_wrapped<(sizeof...(args) > 1)>(Config{}, wrapped_,
                    unwrap_depth_impl<Depth>(
                        Config{}, std::forward<Args>(args)...));
            }

            template <typename... Args>
            auto operator()(Args&&... args) const
                -> decltype(invoke_wrapped(Config{}, std::declval<T const&>(),
                    unwrap_depth_impl<Depth>(Config{},
                        std::forward<Args>(args)...)))
            {
                return invoke_wrapped(Config{}, wrapped_,
                    unwrap_depth_impl<Depth>(
                        Config{}, std::forward<Args>(args)...));
            }
        };

        /// Returns a callable object which unwraps the futures
        /// contained in the given pack args until the depth Depth.
        template <std::size_t Depth, typename Config, typename T>
        auto functional_unwrap_depth_impl(Config, T&& callable)
            -> functional_unwrap_impl<Config, typename std::decay<T>::type,
                Depth>
        {
            return functional_unwrap_impl<Config, typename std::decay<T>::type,
                Depth>(std::forward<T>(callable));
        }
    }
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_UTIL_DETAIL_UNWRAP_IMPL_HPP
