//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//  Copyright (c) 2021 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/copy.hpp

#pragma once

#include <hpx/local/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/modules/futures.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename I1, typename I2>
    struct in_in_result
    {
        HPX_NO_UNIQUE_ADDRESS I1 in1;
        HPX_NO_UNIQUE_ADDRESS I2 in2;

        template <typename II1, typename II2,
            typename Enable = typename std::enable_if<
                std::is_convertible<I1 const&, II1>::value &&
                std::is_convertible<I2 const&, II2>::value>::type>
        constexpr operator in_in_result<II1, II2>() const&
        {
            return {in1, in2};
        }

        template <typename II1, typename II2,
            typename Enable =
                typename std::enable_if<std::is_convertible<I1, II1>::value &&
                    std::is_convertible<I2, II2>::value>::type>
        constexpr operator in_in_result<II1, II2>() &&
        {
            return {std::move(in1), std::move(in2)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in1 & in2;
            // clang-format on
        }
    };

    template <typename I1, typename I2>
    I2 get_in2_element(util::in_in_result<I1, I2>&& p)
    {
        return p.in2;
    }

    template <typename I1, typename I2>
    hpx::future<I2> get_in2_element(hpx::future<util::in_in_result<I1, I2>>&& f)
    {
        return hpx::make_future<I2>(
            std::move(f), [](util::in_in_result<I1, I2>&& p) { return p.in2; });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename O>
    struct in_out_result
    {
        HPX_NO_UNIQUE_ADDRESS I in;
        HPX_NO_UNIQUE_ADDRESS O out;

        template <typename I2, typename O2,
            typename Enable = typename std::enable_if<
                std::is_convertible<I const&, I2>::value &&
                std::is_convertible<O const&, O2>::value>::type>
        constexpr operator in_out_result<I2, O2>() const&
        {
            return {in, out};
        }

        template <typename I2, typename O2,
            typename Enable =
                typename std::enable_if<std::is_convertible<I, I2>::value &&
                    std::is_convertible<O, O2>::value>::type>
        constexpr operator in_out_result<I2, O2>() &&
        {
            return {std::move(in), std::move(out)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in & out;
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename I, typename O>
    std::pair<I, O> get_pair(util::in_out_result<I, O>&& p)
    {
        return std::pair<I, O>{p.in, p.out};
    }

    template <typename I, typename O>
    O get_second_element(util::in_out_result<I, O>&& p)
    {
        return p.out;
    }

    template <typename I, typename O>
    hpx::future<std::pair<I, O>> get_pair(
        hpx::future<util::in_out_result<I, O>>&& f)
    {
        return hpx::make_future<std::pair<I, O>>(
            std::move(f), [](util::in_out_result<I, O>&& p) {
                return std::pair<I, O>{p.in, p.out};
            });
    }

    template <typename I, typename O>
    hpx::future<O> get_second_element(
        hpx::future<util::in_out_result<I, O>>&& f)
    {
        return hpx::make_future<O>(
            std::move(f), [](util::in_out_result<I, O>&& p) { return p.out; });
    }

    // converst a in_out_result into a iterator_range
    template <typename I, typename O>
    hpx::util::iterator_range<I, O> get_subrange(in_out_result<I, O> const& ior)
    {
        return hpx::util::iterator_range<I, O>(ior.in, ior.out);
    }

    template <typename I, typename O>
    hpx::future<hpx::util::iterator_range<I, O>> get_subrange(
        hpx::future<in_out_result<I, O>>&& ior)
    {
        return hpx::make_future<hpx::util::iterator_range<I, O>>(
            std::move(ior), [](in_out_result<I, O>&& ior) {
                return hpx::util::iterator_range<I, O>(ior.in, ior.out);
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename I1, typename I2, typename O>
    struct in_in_out_result
    {
        HPX_NO_UNIQUE_ADDRESS I1 in1;
        HPX_NO_UNIQUE_ADDRESS I2 in2;
        HPX_NO_UNIQUE_ADDRESS O out;

        template <typename II1, typename II2, typename O1,
            typename Enable = typename std::enable_if_t<
                std::is_convertible_v<I1 const&, II1> &&
                std::is_convertible_v<I2 const&, II2> &&
                std::is_convertible_v<O const&, O1>>>
        constexpr operator in_in_out_result<II1, II2, O1>() const&
        {
            return {in1, in2, out};
        }

        template <typename II2, typename II1, typename O1,
            typename Enable = typename std::enable_if_t<
                std::is_convertible_v<I1, II1> &&
                std::is_convertible_v<I2, II2> && std::is_convertible_v<O, O1>>>
        constexpr operator in_in_out_result<II1, II2, O1>() &&
        {
            return {std::move(in1), std::move(in2), std::move(out)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in1 & in2 & out;
            // clang-format on
        }
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename I1, typename I2, typename O>
    O get_third_element(util::in_in_out_result<I1, I2, O>&& p)
    {
        return p.out;
    }

    template <typename I1, typename I2, typename O>
    hpx::future<O> get_third_element(
        hpx::future<util::in_in_out_result<I1, I2, O>>&& f)
    {
        return lcos::make_future<O>(std::move(f),
            [](in_in_out_result<I1, I2, O>&& p) { return p.out; });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename O1, typename O2>
    struct in_out_out_result
    {
        HPX_NO_UNIQUE_ADDRESS I in;
        HPX_NO_UNIQUE_ADDRESS O1 out1;
        HPX_NO_UNIQUE_ADDRESS O2 out2;

        template <typename II, typename OO1, typename OO2,
            typename Enable =
                typename std::enable_if_t<std::is_convertible_v<I const&, II> &&
                    std::is_convertible_v<O1 const&, OO1> &&
                    std::is_convertible_v<O2 const&, OO2>>>
        constexpr operator in_out_out_result<II, OO1, OO2>() const&
        {
            return {in, out1, out2};
        }

        template <typename II, typename OO1, typename OO2,
            typename Enable =
                typename std::enable_if_t<std::is_convertible_v<I, II> &&
                    std::is_convertible_v<O1, OO1> &&
                    std::is_convertible_v<O2, OO2>>>
        constexpr operator in_out_out_result<II, OO1, OO2>() &&
        {
            return {std::move(in), std::move(out1), std::move(out2)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in & out1 & out2;
            // clang-format on
        }
    };

    template <typename... Ts>
    constexpr HPX_FORCEINLINE in_out_out_result<Ts...> make_in_out_out_result(
        hpx::tuple<Ts...>&& t)
    {
        static_assert(hpx::tuple_size<hpx::tuple<Ts...>>::value == 3,
            "size of tuple should be 3 to convert to in_out_out_result");

        using result_type = in_out_out_result<Ts...>;

        return result_type{hpx::get<0>(t), hpx::get<1>(t), hpx::get<2>(t)};
    }

    template <typename... Ts>
    hpx::future<in_out_out_result<Ts...>> make_in_out_out_result(
        hpx::future<hpx::tuple<Ts...>>&& f)
    {
        static_assert(hpx::tuple_size<hpx::tuple<Ts...>>::value == 3,
            "size of tuple should be 3 to convert to in_out_out_result");

        using result_type = in_out_out_result<Ts...>;

        return lcos::make_future<result_type>(
            std::move(f), [](hpx::tuple<Ts...>&& t) -> result_type {
                return make_in_out_out_result(std::move(t));
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename I, typename F>
    struct in_fun_result
    {
        HPX_NO_UNIQUE_ADDRESS I in;
        HPX_NO_UNIQUE_ADDRESS F fun;

        template <typename I2, typename F2,
            typename Enable = typename std::enable_if<
                std::is_convertible<I const&, I2>::value &&
                std::is_convertible<F const&, F2>::value>::type>
        constexpr operator in_fun_result<I2, F2>() const&
        {
            return {in, fun};
        }

        template <typename I2, typename F2,
            typename Enable =
                typename std::enable_if<std::is_convertible<I, I2>::value &&
                    std::is_convertible<F, F2>::value>::type>
        constexpr operator in_fun_result<I2, F2>() &&
        {
            return {std::move(in), std::move(fun)};
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned)
        {
            // clang-format off
            ar & in & fun;
            // clang-format on
        }
    };

    template <typename Iterator, typename Sentinel = Iterator>
    hpx::util::iterator_range<Iterator, Sentinel> make_subrange(
        Iterator iterator, Sentinel sentinel)
    {
        return hpx::util::make_iterator_range<Iterator, Sentinel>(
            iterator, sentinel);
    }

    template <typename Iterator, typename Sentinel = Iterator>
    hpx::future<hpx::util::iterator_range<Iterator, Sentinel>> make_subrange(
        hpx::future<Iterator>&& iterator, Sentinel sentinel)
    {
        return lcos::make_future<hpx::util::iterator_range<Iterator, Sentinel>>(
            std::move(iterator), [sentinel](Iterator&& it) {
                return hpx::util::iterator_range<Iterator, Sentinel>(
                    it, sentinel);
            });
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename ZipIter>
        in_out_result<typename hpx::tuple_element<0,
                          typename ZipIter::iterator_tuple_type>::type,
            typename hpx::tuple_element<1,
                typename ZipIter::iterator_tuple_type>::type>
        get_in_out_result(ZipIter&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_out_result<
                typename hpx::tuple_element<0, iterator_tuple_type>::type,
                typename hpx::tuple_element<1, iterator_tuple_type>::type>;

            iterator_tuple_type t = zipiter.get_iterator_tuple();
            return result_type{hpx::get<0>(t), hpx::get<1>(t)};
        }

        template <typename ZipIter>
        hpx::future<
            in_out_result<typename hpx::tuple_element<0,
                              typename ZipIter::iterator_tuple_type>::type,
                typename hpx::tuple_element<1,
                    typename ZipIter::iterator_tuple_type>::type>>
        get_in_out_result(hpx::future<ZipIter>&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_out_result<
                typename hpx::tuple_element<0, iterator_tuple_type>::type,
                typename hpx::tuple_element<1, iterator_tuple_type>::type>;

            return lcos::make_future<result_type>(
                std::move(zipiter), [](ZipIter zipiter) {
                    return get_in_out_result(std::move(zipiter));
                });
        }

        template <typename ZipIter>
        in_in_out_result<typename hpx::tuple_element<0,
                             typename ZipIter::iterator_tuple_type>::type,
            typename hpx::tuple_element<1,
                typename ZipIter::iterator_tuple_type>::type,
            typename hpx::tuple_element<2,
                typename ZipIter::iterator_tuple_type>::type>
        get_in_in_out_result(ZipIter&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_in_out_result<
                typename hpx::tuple_element<0, iterator_tuple_type>::type,
                typename hpx::tuple_element<1, iterator_tuple_type>::type,
                typename hpx::tuple_element<2, iterator_tuple_type>::type>;

            iterator_tuple_type t = zipiter.get_iterator_tuple();
            return result_type{hpx::get<0>(t), hpx::get<1>(t), hpx::get<2>(t)};
        }

        template <typename ZipIter>
        hpx::future<
            in_in_out_result<typename hpx::tuple_element<0,
                                 typename ZipIter::iterator_tuple_type>::type,
                typename hpx::tuple_element<1,
                    typename ZipIter::iterator_tuple_type>::type,
                typename hpx::tuple_element<2,
                    typename ZipIter::iterator_tuple_type>::type>>
        get_in_in_out_result(hpx::future<ZipIter>&& zipiter)
        {
            using iterator_tuple_type = typename ZipIter::iterator_tuple_type;

            using result_type = in_in_out_result<
                typename hpx::tuple_element<0, iterator_tuple_type>::type,
                typename hpx::tuple_element<1, iterator_tuple_type>::type,
                typename hpx::tuple_element<2, iterator_tuple_type>::type>;

            return lcos::make_future<result_type>(
                std::move(zipiter), [](ZipIter zipiter) {
                    return get_in_in_out_result(std::move(zipiter));
                });
        }
    }    // namespace detail
}}}      // namespace hpx::parallel::util

namespace hpx { namespace ranges {
    using hpx::parallel::util::in_fun_result;
    using hpx::parallel::util::in_in_out_result;
    using hpx::parallel::util::in_in_result;
    using hpx::parallel::util::in_out_out_result;
    using hpx::parallel::util::in_out_result;
}}    // namespace hpx::ranges
