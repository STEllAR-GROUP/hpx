#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>

namespace hpx::parallel { namespace detail {

    template <typename _T>
    struct fold_left : public algorithm<fold_left<_T>, _T>
    {
        constexpr fold_left() noexcept
          : algorithm<fold_left, _T>("fold_left")
        {
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        HPX_HOST_DEVICE static constexpr T sequential(
            ExPolicy&&, FwdIter first, Sent last, T&& init, F&& f)
        {
            hpx::reduce(first, last, HPX_FORWARD(T, init), HPX_FORWARD(F, f));
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F>
        static util::detail::algorithm_result_t<ExPolicy, FwdIter> parallel(
            ExPolicy&& policy, FwdIter first, Sent last, T&& init, F&& f)
        {
            hpx::reduce(HPX_FORWARD(ExPolicy, policy), first, last, HPX_FORWARD(T, init),
                HPX_FORWARD(F, f));
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
            FwdIter last, T&& init, F&& f)
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
    // clang-format on
    friend T tag_fallback_invoke(
        fold_left_t, FwdIter first, FwdIter last, T&& init, F&& f)
    {
        static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
            "Requires at least forward iterator.");

        return hpx::parallel::detail::fold_left<T>().call(hpx::execution::seq,
            first, last, HPX_FORWARD(T, init), HPX_FORWARD(F, f));
    }
} fold_left{};
}    // namespace hpx
