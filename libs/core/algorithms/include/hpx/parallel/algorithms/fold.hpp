#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>


#include <iterator>
#include <optional>

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
            hpx::reduce(first, last, init, f);
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        static util::detail::algorithm_result_t<ExPolicy, FwdIter> parallel(
            ExPolicy&& policy, FwdIter first, Sent last, T&& init, F&& f)
        {
            hpx::reduce(HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(T, init), HPX_FORWARD(F, f));
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
    friend
        typename hpx::parallel::util::detail::algorithm_result<ExPolicy>::type
        tag_fallback_invoke(fold_left_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, T init, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        using result_type =
            typename hpx::parallel::util::detail::algorithm_result<
                ExPolicy>::type;

        return hpx::util::void_guard<result_type>(),
               hpx::parallel::detail::fold_left<FwdIter>().call(
                   HPX_FORWARD(ExPolicy, policy), first, last,
                   HPX_FORWARD(T, init), HPX_FORWARD(F, f));
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
            using U = decltype(hpx::fold_left(
                std::move(first), last, ::hpx::traits::iter_value_t<FwdIter>(*first), f));
            if (first == last)
                return std::optional<U>();

            ::hpx::traits::iter_value_t<FwdIter> init = *first;
            std::advance(first, 1);
            return hpx::fold_left(first, last, init, HPX_FORWARD(F, f));
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename F>
        static util::detail::algorithm_result_t<ExPolicy, FwdIter> parallel(
            ExPolicy&& policy, FwdIter first, Sent last, F&& f)
        {
            using U = decltype(hpx::fold_left(
                std::move(first), last, ::hpx::traits::iter_value_t<FwdIter>(*first), f));
            if (first == last)
                return std::optional<U>();

            ::hpx::traits::iter_value_t<FwdIter> init = *first;
            std::advance(first, 1);
            return hpx::fold_left(HPX_FORWARD(ExPolicy, policy), first, last,
                init, HPX_FORWARD(F, f));
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
    friend
        typename hpx::parallel::util::detail::algorithm_result<ExPolicy>::type
        tag_fallback_invoke(fold_left_first_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        using result_type =
            typename hpx::parallel::util::detail::algorithm_result<
                ExPolicy>::type;

        return hpx::util::void_guard<result_type>(),
               hpx::parallel::detail::fold_left_first<FwdIter>().call(
                   HPX_FORWARD(ExPolicy, policy), first, last,
                   HPX_FORWARD(F, f));
    }

    template <typename FwdIter, typename F>    // TODO : add concept
    friend auto tag_fallback_invoke(
        fold_left_first_t, FwdIter first, FwdIter last, F f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        using U = decltype(hpx::fold_left(
            std::move(first), last, ::hpx::traits::iter_value_t<FwdIter>(*first), f));

        return hpx::parallel::detail::fold_left_first<std::optional<U>>().call(
            hpx::execution::seq, first, last, HPX_FORWARD(F, f));
    }
} fold_left_first{};
}    // namespace hpx
