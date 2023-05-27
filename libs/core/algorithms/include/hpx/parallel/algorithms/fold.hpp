#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>

#include <iterator>
#include <numeric>

namespace hpx::parallel { namespace detail {

    template <typename T_>
    struct fold_left : public algorithm<fold_left<T_>, T_>
    {
        constexpr fold_left() noexcept
          : algorithm<fold_left, T_>("fold_left")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        HPX_HOST_DEVICE static constexpr T sequential(
            ExPolicy&&, FwdIter first, Sent last, T&& init, F&& f)
        {
            if (first == last)
                return HPX_MOVE(init);
            T acc = HPX_MOVE(init);
            while (first != last)
                acc = HPX_MOVE(f(HPX_MOVE(acc), *first++));
            return HPX_MOVE(acc);
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        static constexpr auto parallel(
            ExPolicy&& policy, FwdIter first, Sent last, T&& init, F&& f)
        {
#ifdef HPX_WITH_CXX17_STD_EXECUTION_POLICES
            return std::reduce(HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(T, init), HPX_FORWARD(F, f));
#else
            return std::reduce(
                first, last, HPX_FORWARD(T, init), HPX_FORWARD(F, f));
#endif
        }
    };

}}    // namespace hpx::parallel::detail

namespace hpx {
inline constexpr struct fold_left_t final
  : hpx::detail::tag_parallel_algorithm<fold_left_t>
{
private:
    template <typename ExPolicy, typename FwdIter, typename T,
        typename F>    // TODO : add concept
    friend T tag_fallback_invoke(fold_left_t, ExPolicy&& policy, FwdIter first,
        FwdIter last, T init, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        return hpx::parallel::detail::fold_left<T>().call(
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(T, init),
            HPX_FORWARD(F, f));
    }

    template <typename FwdIter, typename T,
        typename F>    // TODO : add concept
    friend T tag_fallback_invoke(
        fold_left_t, FwdIter first, FwdIter last, T init, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        return hpx::parallel::detail::fold_left<T>().call(hpx::execution::seq,
            first, last, HPX_FORWARD(T, init), HPX_FORWARD(F, f));
    }
} fold_left{};
}    // namespace hpx

namespace hpx::parallel { namespace detail {

    template <typename T_>
    struct fold_left_first : public algorithm<fold_left_first<T_>, T_>
    {
        constexpr fold_left_first() noexcept
          : algorithm<fold_left_first, T_>("fold_left_first")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F>
        HPX_HOST_DEVICE static constexpr auto sequential(
            ExPolicy&&, FwdIter first, Sent last, F&& f)
        {
            using T = ::hpx::traits::iter_value_t<FwdIter>;
            using U =
                decltype(hpx::fold_left(HPX_MOVE(first), last, T(*first), f));

            if (first == last)
                return hpx::optional<U>();

            T init = *first;

            std::advance(first, 1);

            return hpx::optional<U>(
                hpx::parallel::detail::fold_left<T>().call(hpx::execution::seq,
                    first, last, HPX_FORWARD(T, init), HPX_FORWARD(F, f)));
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F>
        static constexpr auto parallel(
            ExPolicy&& policy, FwdIter first, Sent last, F&& f)
        {
            using T = ::hpx::traits::iter_value_t<FwdIter>;
            using U =
                decltype(hpx::fold_left(HPX_MOVE(first), last, T(*first), f));

            if (first == last)
                return hpx::optional<U>();

            T init = *first;

            std::advance(first, 1);

            return hpx::optional<U>(hpx::parallel::detail::fold_left<T>().call(
                HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(T, init), HPX_FORWARD(F, f)));
        }
    };

}}    // namespace hpx::parallel::detail

namespace hpx {
inline constexpr struct fold_left_first_t final
  : hpx::detail::tag_parallel_algorithm<fold_left_first_t>
{
private:
    template <typename ExPolicy, typename FwdIter,
        typename F>    // TODO : add concept
    friend auto tag_fallback_invoke(
        fold_left_first_t, ExPolicy&& policy, FwdIter first, FwdIter last, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        using result_type =
            typename hpx::parallel::util::detail::algorithm_result<
                ExPolicy>::type;

        using U = decltype(hpx::fold_left(HPX_MOVE(first), last,
            ::hpx::traits::iter_value_t<FwdIter>(*first), f));

        return hpx::parallel::detail::fold_left_first<hpx::optional<U>>().call(
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(F, f));
    }

    template <typename FwdIter, typename F>    // TODO : add concept
    friend auto tag_fallback_invoke(
        fold_left_first_t, FwdIter first, FwdIter last, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        using U = decltype(hpx::fold_left(HPX_MOVE(first), last,
            ::hpx::traits::iter_value_t<FwdIter>(*first), f));

        return hpx::parallel::detail::fold_left_first<hpx::optional<U>>().call(
            hpx::execution::seq, first, last, HPX_FORWARD(F, f));
    }
} fold_left_first{};
}    // namespace hpx

namespace hpx::parallel { namespace detail {

    template <typename T_>
    struct fold_right : public algorithm<fold_right<T_>, T_>
    {
        constexpr fold_right() noexcept
          : algorithm<fold_right, T_>("fold_right")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        HPX_HOST_DEVICE static constexpr auto sequential(
            ExPolicy&&, FwdIter first, Sent last, T&& init, F&& f)
        {
            using U = std::decay_t<
                std::invoke_result_t<F&, hpx::traits::iter_reference_t<FwdIter>, T>>;
            if (first == last)
                return U(HPX_MOVE(init));
     
            U accum = f(*--last, HPX_MOVE(init));
            while (first != last)
                accum = f(*--last, HPX_MOVE(accum));
            return accum;
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        static constexpr auto parallel(
            ExPolicy&& policy, FwdIter first, Sent last, T&& init, F&& f)
        {
            exit(
                1);    // parallel version of fold_right has not been implemented
            return f(first, init);
        }
    };

}}    // namespace hpx::parallel::detail

namespace hpx {
inline constexpr struct fold_right_t final
  : hpx::detail::tag_parallel_algorithm<fold_right_t>
{
private:
    template <typename ExPolicy, typename FwdIter, typename T,
        typename F>    // TODO : add concept
    friend T tag_fallback_invoke(fold_right_t, ExPolicy&& policy, FwdIter first,
        FwdIter last, T init, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        return hpx::parallel::detail::fold_right<T>().call(
            HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(T, init),
            HPX_FORWARD(F, f));
    }

    template <typename FwdIter, typename T,
        typename F>    // TODO : add concept
    friend T tag_fallback_invoke(
        fold_right_t, FwdIter first, FwdIter last, T init, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        return hpx::parallel::detail::fold_right<T>().call(hpx::execution::seq,
            first, last, HPX_FORWARD(T, init), HPX_FORWARD(F, f));
    }
} fold_right{};
}    // namespace hpx
