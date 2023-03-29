//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel::util {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct loop
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Begin call(
                Begin it, End end, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    HPX_INVOKE(f, it);
                }
                return it;
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Begin call(
                Begin it, End end, CancelToken& tok, F&& f)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return it;

                return call(it, end, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    struct loop_t final : hpx::functional::detail::tag_fallback<loop_t>
    {
    private:
        template <typename ExPolicy, typename Begin, typename End, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin
        tag_fallback_invoke(hpx::parallel::util::loop_t, ExPolicy&&,
            Begin begin, End end, F&& f)
        {
            return detail::loop<Begin>::call(begin, end, HPX_FORWARD(F, f));
        }

        template <typename ExPolicy, typename Begin, typename End,
            typename CancelToken, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin
        tag_fallback_invoke(hpx::parallel::util::loop_t, ExPolicy&&,
            Begin begin, End end, CancelToken& tok, F&& f)
        {
            return detail::loop<Begin>::call(
                begin, end, tok, HPX_FORWARD(F, f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    inline constexpr loop_t loop = loop_t{};
#else
    template <typename ExPolicy, typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin loop(
        ExPolicy&& policy, Begin begin, End end, F&& f)
    {
        return hpx::parallel::util::loop_t{}(
            HPX_FORWARD(ExPolicy, policy), begin, end, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Begin, typename End,
        typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin loop(
        ExPolicy&& policy, Begin begin, End end, CancelToken& tok, F&& f)
    {
        return hpx::parallel::util::loop_t{}(
            HPX_FORWARD(ExPolicy, policy), begin, end, tok, HPX_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function starting from a given
        // iterator position till the predicate returns true.
        template <typename Iterator>
        struct loop_pred
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Begin, typename End, typename Pred>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Begin call(
                Begin it, End end, Pred&& pred)
            {
                for (/**/; it != end; ++it)
                {
                    if (HPX_INVOKE(pred, it))
                        return it;
                }
                return it;
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct loop_pred_t final
      : hpx::functional::detail::tag_fallback<loop_pred_t<ExPolicy>>
    {
    private:
        template <typename Begin, typename End, typename Pred>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin
        tag_fallback_invoke(hpx::parallel::util::loop_pred_t<ExPolicy>,
            Begin begin, End end, Pred&& pred)
        {
            return detail::loop_pred<Begin>::call(
                begin, end, HPX_FORWARD(Pred, pred));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_pred_t<ExPolicy> loop_pred = loop_pred_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Begin, typename End, typename Pred>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin loop_pred(
        Begin begin, End end, Pred&& pred)
    {
        return hpx::parallel::util::loop_pred_t<ExPolicy>{}(
            begin, end, HPX_FORWARD(Pred, pred));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iterator>
        struct loop_ind
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Begin, typename End, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Begin call(
                Begin it, End end, F&& f)
            {
                for (/**/; it != end; ++it)
                {
                    HPX_INVOKE(f, *it);
                }
                return it;
            }

            template <typename Begin, typename End, typename CancelToken,
                typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Begin call(
                Begin it, End end, CancelToken& tok, F&& f)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return it;

                return call(it, end, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct loop_ind_t final
      : hpx::functional::detail::tag_fallback<loop_ind_t<ExPolicy>>
    {
    private:
        template <typename Begin, typename End, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin
        tag_fallback_invoke(hpx::parallel::util::loop_ind_t<ExPolicy>,
            Begin begin, End end, F&& f)
        {
            return detail::loop_ind<std::decay_t<Begin>>::call(
                begin, end, HPX_FORWARD(F, f));
        }

        template <typename Begin, typename End, typename CancelToken,
            typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin
        tag_fallback_invoke(hpx::parallel::util::loop_ind_t<ExPolicy>,
            Begin begin, End end, CancelToken& tok, F&& f)
        {
            return detail::loop_ind<std::decay_t<Begin>>::call(
                begin, end, tok, HPX_FORWARD(F, f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_ind_t loop_ind = loop_ind_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Begin, typename End, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin loop_ind(
        Begin begin, End end, F&& f)
    {
        return hpx::parallel::util::loop_ind_t<ExPolicy>{}(
            begin, end, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Begin, typename End,
        typename CancelToken, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Begin loop_ind(
        Begin begin, End end, CancelToken& tok, F&& f)
    {
        return hpx::parallel::util::loop_ind_t<ExPolicy>{}(
            begin, end, tok, HPX_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function starting from a given
        // iterator position.
        template <typename Iter1, typename Iter2>
        struct loop2
        {
            ///////////////////////////////////////////////////////////////////
            template <typename Begin1, typename End1, typename Begin2,
                typename F>
            HPX_HOST_DEVICE
                HPX_FORCEINLINE static constexpr std::pair<Begin1, Begin2>
                call(Begin1 it1, End1 end1, Begin2 it2, F&& f)
            {
                for (/**/; it1 != end1; (void) ++it1, ++it2)
                {
                    HPX_INVOKE(f, it1, it2);
                }

                return std::make_pair(HPX_MOVE(it1), HPX_MOVE(it2));
            }
        };
    }    // namespace detail

    template <typename ExPolicy>
    struct loop2_t final
      : hpx::functional::detail::tag_fallback<loop2_t<ExPolicy>>
    {
    private:
        template <typename Begin1, typename End1, typename Begin2, typename F>
        friend HPX_HOST_DEVICE
            HPX_FORCEINLINE constexpr std::pair<Begin1, Begin2>
            tag_fallback_invoke(hpx::parallel::util::loop2_t<ExPolicy>,
                Begin1 begin1, End1 end1, Begin2 begin2, F&& f)
        {
            return detail::loop2<Begin1, Begin2>::call(
                begin1, end1, begin2, HPX_FORWARD(F, f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop2_t<ExPolicy> loop2 = loop2_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Begin1, typename End1,
        typename Begin2, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr std::pair<Begin1, Begin2> loop2(
        Begin1 begin1, End1 end1, Begin2 begin2, F&& f)
    {
        return hpx::parallel::util::loop2_t<ExPolicy>{}(
            begin1, end1, begin2, HPX_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        struct loop_n_helper
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    HPX_INVOKE(f, it);
                    HPX_INVOKE(f, ++it);
                    HPX_INVOKE(f, ++it);
                    HPX_INVOKE(f, ++it);
                }
                for (/**/; count < num; (void) ++count, ++it)
                {
                    HPX_INVOKE(f, it);
                }
                return it;
            }

            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::true_type)
            {
                while (num >= 4)    //-V112
                {
                    HPX_INVOKE(f, it);
                    HPX_INVOKE(f, it + 1);
                    HPX_INVOKE(f, it + 2);
                    HPX_INVOKE(f, it + 3);

                    it += 4;     //-V112
                    num -= 4;    //-V112
                }

                switch (num)
                {
                case 3:
                    HPX_INVOKE(f, it);
                    HPX_INVOKE(f, it + 1);
                    HPX_INVOKE(f, it + 2);
                    break;

                case 2:
                    HPX_INVOKE(f, it);
                    HPX_INVOKE(f, it + 1);
                    break;

                case 1:
                    HPX_INVOKE(f, it);
                    break;

                default:
                    break;
                }

                return static_cast<Iter>(it + num);
            }

            template <typename Iter, typename CancelToken, typename F,
                typename Tag>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
                Iter it, std::size_t num, CancelToken& tok, F&& f, Tag tag)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return it;

                return call(it, num, HPX_FORWARD(F, f), tag);
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct loop_n_t final
      : hpx::functional::detail::tag_fallback<loop_n_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter
        tag_fallback_invoke(hpx::parallel::util::loop_n_t<ExPolicy>, Iter it,
            std::size_t count, F&& f)
        {
            using pred = std::integral_constant<bool,
                hpx::traits::is_random_access_iterator_v<Iter> ||
                    std::is_integral_v<Iter>>;

            return detail::loop_n_helper::call(
                it, count, HPX_FORWARD(F, f), pred());
        }

        template <typename Iter, typename CancelToken, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE Iter tag_fallback_invoke(
            hpx::parallel::util::loop_n_t<ExPolicy>, Iter it, std::size_t count,
            CancelToken& tok, F&& f)
        {
            using pred = std::integral_constant<bool,
                hpx::traits::is_random_access_iterator_v<Iter> ||
                    std::is_integral_v<Iter>>;

            return detail::loop_n_helper::call(
                it, count, tok, HPX_FORWARD(F, f), pred());
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_n_t<ExPolicy> loop_n = loop_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter loop_n(
        Iter it, std::size_t count, F&& f)
    {
        return hpx::parallel::util::loop_n_t<ExPolicy>{}(
            it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter loop_n(
        Iter it, std::size_t count, CancelToken& tok, F&& f)
    {
        return hpx::parallel::util::loop_n_t<ExPolicy>{}(
            it, count, tok, HPX_FORWARD(F, f));
    }
#endif

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        struct loop_n_ind_helper
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::false_type)
            {
                std::size_t count(num & std::size_t(-4));    // -V112
                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    HPX_INVOKE(f, *it);
                    HPX_INVOKE(f, *++it);
                    HPX_INVOKE(f, *++it);
                    HPX_INVOKE(f, *++it);
                }
                for (/**/; count < num; (void) ++count, ++it)
                {
                    HPX_INVOKE(f, *it);
                }

                return it;
            }

            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                Iter it, std::size_t num, F&& f, std::true_type)
            {
                while (num >= 4)    //-V112
                {
                    HPX_INVOKE(f, *it);
                    HPX_INVOKE(f, *(it + 1));
                    HPX_INVOKE(f, *(it + 2));
                    HPX_INVOKE(f, *(it + 3));

                    it += 4;     //-V112
                    num -= 4;    //-V112
                }

                switch (num)
                {
                case 3:
                    HPX_INVOKE(f, *it);
                    HPX_INVOKE(f, *(it + 1));
                    HPX_INVOKE(f, *(it + 2));
                    break;

                case 2:
                    HPX_INVOKE(f, *it);
                    HPX_INVOKE(f, *(it + 1));
                    break;

                case 1:
                    HPX_INVOKE(f, *it);
                    break;

                default:
                    break;
                }

                return static_cast<Iter>(it + num);
            }

            template <typename Iter, typename CancelToken, typename F,
                typename Tag>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
                Iter it, std::size_t num, CancelToken& tok, F&& f, Tag tag)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return it;

                return call(it, num, HPX_FORWARD(F, f), tag);
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct loop_n_ind_t final
      : hpx::functional::detail::tag_fallback<loop_n_ind_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter
        tag_fallback_invoke(hpx::parallel::util::loop_n_ind_t<ExPolicy>,
            Iter it, std::size_t count, F&& f)
        {
            using pred = std::integral_constant<bool,
                hpx::traits::is_random_access_iterator_v<Iter> ||
                    std::is_integral<Iter>::value>;

            return detail::loop_n_ind_helper::call(
                it, count, HPX_FORWARD(F, f), pred());
        }

        template <typename Iter, typename CancelToken, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter
        tag_fallback_invoke(hpx::parallel::util::loop_n_ind_t<ExPolicy>,
            Iter it, std::size_t count, CancelToken& tok, F&& f)
        {
            using pred = std::integral_constant<bool,
                hpx::traits::is_random_access_iterator_v<Iter> ||
                    std::is_integral<Iter>::value>;

            return detail::loop_n_ind_helper::call(
                it, count, tok, HPX_FORWARD(F, f), pred());
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_n_ind_t<ExPolicy> loop_n_ind =
        loop_n_ind_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter loop_n_ind(
        Iter it, std::size_t count, F&& f)
    {
        return hpx::parallel::util::loop_n_ind_t<ExPolicy>{}(
            it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter loop_n_ind(
        Iter it, std::size_t count, CancelToken& tok, F&& f)
    {
        return hpx::parallel::util::loop_n_ind_t<ExPolicy>{}(
            it, count, tok, HPX_FORWARD(F, f));
    }
#endif

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position. If an exception is thrown,
        // the given cleanup function will be called.
        template <typename IterCat>
        struct loop_with_cleanup
        {
            ///////////////////////////////////////////////////////////////////
            template <typename FwdIter, typename F, typename Cleanup>
            static FwdIter call(
                FwdIter it, FwdIter last, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = it;
                try
                {
                    for (/**/; it != last; ++it)
                    {
                        HPX_INVOKE(f, it);
                    }
                    return it;
                }
                catch (...)
                {
                    for (/**/; base != it; ++base)
                        cleanup(base);
                    throw;
                }
            }

            template <typename Iter, typename FwdIter, typename F,
                typename Cleanup>
            static FwdIter call(
                Iter it, Iter last, FwdIter dest, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = dest;
                try
                {
                    for (/**/; it != last; (void) ++it, ++dest)
                        f(it, dest);
                    return dest;
                }
                catch (...)
                {
                    for (/**/; base != dest; ++base)
                    {
                        HPX_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F, typename Cleanup>
    HPX_FORCEINLINE constexpr Iter loop_with_cleanup(
        Iter it, Iter last, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup<cat>::call(
            it, last, HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename FwdIter, typename F, typename Cleanup>
    HPX_FORCEINLINE constexpr FwdIter loop_with_cleanup(
        Iter it, Iter last, FwdIter dest, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup<cat>::call(
            it, last, dest, HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct loop_with_cleanup_n
        {
            ///////////////////////////////////////////////////////////////////
            template <typename FwdIter, typename F, typename Cleanup>
            static FwdIter call(
                FwdIter it, std::size_t num, F&& f, Cleanup&& cleanup)
            {
                FwdIter base = it;
                try
                {
                    std::size_t count(num & std::size_t(-4));    // -V112
                    for (std::size_t i = 0; i < count;
                         (void) ++it, i += 4)    // -V112
                    {
                        HPX_INVOKE(f, it);
                        HPX_INVOKE(f, ++it);
                        HPX_INVOKE(f, ++it);
                        HPX_INVOKE(f, ++it);
                    }
                    for (/**/; count < num; (void) ++count, ++it)
                    {
                        HPX_INVOKE(f, it);
                    }
                    return it;
                }
                catch (...)
                {
                    for (/**/; base != it; ++base)
                    {
                        HPX_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }

            template <typename Iter, typename FwdIter, typename F,
                typename Cleanup>
            static FwdIter call(Iter it, std::size_t num, FwdIter dest, F&& f,
                Cleanup&& cleanup)
            {
                FwdIter base = dest;
                try
                {
                    std::size_t count(num & std::size_t(-4));    // -V112
                    for (std::size_t i = 0; i < count;
                         (void) ++it, ++dest, i += 4)    // -V112
                    {
                        HPX_INVOKE(f, it, dest);
                        HPX_INVOKE(f, ++it, ++dest);
                        HPX_INVOKE(f, ++it, ++dest);
                        HPX_INVOKE(f, ++it, ++dest);
                    }
                    for (/**/; count < num; (void) ++count, ++it, ++dest)
                    {
                        HPX_INVOKE(f, it, dest);
                    }
                    return dest;
                }
                catch (...)
                {
                    for (/**/; base != dest; ++base)
                    {
                        HPX_INVOKE(cleanup, base);
                    }
                    throw;
                }
            }

            template <typename FwdIter, typename CancelToken, typename F,
                typename Cleanup>
            HPX_HOST_DEVICE HPX_FORCEINLINE static FwdIter call_with_token(
                FwdIter it, std::size_t num, CancelToken& tok, F&& f,
                Cleanup&& cleanup)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return it;

                return call(
                    it, num, HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
            }

            template <typename Iter, typename FwdIter, typename CancelToken,
                typename F, typename Cleanup>
            static FwdIter call_with_token(Iter it, std::size_t num,
                FwdIter dest, CancelToken& tok, F&& f, Cleanup&& cleanup)
            {
                // check at the start of a partition only
                if (tok.was_cancelled())
                    return dest;

                return call(it, num, dest, HPX_FORWARD(F, f),
                    HPX_FORWARD(Cleanup, cleanup));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename F, typename Cleanup>
    HPX_FORCEINLINE constexpr Iter loop_with_cleanup_n(
        Iter it, std::size_t count, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call(
            it, count, HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename FwdIter, typename F, typename Cleanup>
    HPX_FORCEINLINE constexpr FwdIter loop_with_cleanup_n(
        Iter it, std::size_t count, FwdIter dest, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call(
            it, count, dest, HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename CancelToken, typename F, typename Cleanup>
    HPX_FORCEINLINE constexpr Iter loop_with_cleanup_n_with_token(
        Iter it, std::size_t count, CancelToken& tok, F&& f, Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call_with_token(
            it, count, tok, HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
    }

    template <typename Iter, typename FwdIter, typename CancelToken, typename F,
        typename Cleanup>
    HPX_FORCEINLINE constexpr FwdIter loop_with_cleanup_n_with_token(Iter it,
        std::size_t count, FwdIter dest, CancelToken& tok, F&& f,
        Cleanup&& cleanup)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::loop_with_cleanup_n<cat>::call_with_token(it, count,
            dest, tok, HPX_FORWARD(F, f), HPX_FORWARD(Cleanup, cleanup));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct loop_idx_n
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t num, F&& f)
            {
                std::size_t count(num & std::size_t(-4));    // -V112

                for (std::size_t i = 0; i < count;
                     (void) ++it, i += 4)    // -V112
                {
                    HPX_INVOKE(f, *it, base_idx++);
                    HPX_INVOKE(f, *++it, base_idx++);
                    HPX_INVOKE(f, *++it, base_idx++);
                    HPX_INVOKE(f, *++it, base_idx++);
                }
                for (/**/; count < num; (void) ++count, ++it, ++base_idx)
                {
                    HPX_INVOKE(f, *it, base_idx);
                }
                return it;
            }

            template <typename Iter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t count,
                CancelToken& tok, F&& f)
            {
                if (tok.was_cancelled(base_idx))
                    return it;

                return call(base_idx, it, count, HPX_FORWARD(F, f));
            }
        };

        template <>
        struct loop_idx_n<std::random_access_iterator_tag>
        {
            ///////////////////////////////////////////////////////////////////
            // handle sequences of non-futures
            template <typename Iter, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static constexpr Iter call(
                std::size_t base_idx, Iter it, std::size_t num, F&& f)
            {
                while (num >= 4)    //-V112
                {
                    HPX_INVOKE(f, *it, base_idx++);
                    HPX_INVOKE(f, *(it + 1), base_idx++);
                    HPX_INVOKE(f, *(it + 2), base_idx++);
                    HPX_INVOKE(f, *(it + 3), base_idx++);

                    it += 4;     //-V112
                    num -= 4;    //-V112
                }

                switch (num)
                {
                case 3:
                    HPX_INVOKE(f, *it, base_idx++);
                    HPX_INVOKE(f, *(it + 1), base_idx++);
                    HPX_INVOKE(f, *(it + 2), base_idx++);
                    break;

                case 2:
                    HPX_INVOKE(f, *it, base_idx++);
                    HPX_INVOKE(f, *(it + 1), base_idx++);
                    break;

                case 1:
                    HPX_INVOKE(f, *it, base_idx);
                    break;

                default:
                    break;
                }

                return static_cast<Iter>(it + num);
            }

            template <typename Iter, typename CancelToken, typename F>
            HPX_HOST_DEVICE HPX_FORCEINLINE static Iter call(
                std::size_t base_idx, Iter it, std::size_t num,
                CancelToken& tok, F&& f)
            {
                if (tok.was_cancelled(base_idx))
                    return it;

                return call(base_idx, it, num, HPX_FORWARD(F, f));
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////
    template <typename ExPolicy>
    struct loop_idx_n_t final
      : hpx::functional::detail::tag_fallback<loop_idx_n_t<ExPolicy>>
    {
    private:
        template <typename Iter, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter
        tag_fallback_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
            std::size_t base_idx, Iter it, std::size_t count, F&& f)
        {
            using cat = typename std::iterator_traits<Iter>::iterator_category;
            return detail::loop_idx_n<cat>::call(
                base_idx, it, count, HPX_FORWARD(F, f));
        }

        template <typename Iter, typename CancelToken, typename F>
        friend HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter
        tag_fallback_invoke(hpx::parallel::util::loop_idx_n_t<ExPolicy>,
            std::size_t base_idx, Iter it, std::size_t count, CancelToken& tok,
            F&& f)
        {
            using cat = typename std::iterator_traits<Iter>::iterator_category;
            return detail::loop_idx_n<cat>::call(
                base_idx, it, count, tok, HPX_FORWARD(F, f));
        }
    };

#if !defined(HPX_COMPUTE_DEVICE_CODE)
    template <typename ExPolicy>
    inline constexpr loop_idx_n_t<ExPolicy> loop_idx_n =
        loop_idx_n_t<ExPolicy>{};
#else
    template <typename ExPolicy, typename Iter, typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter loop_idx_n(
        std::size_t base_idx, Iter it, std::size_t count, F&& f)
    {
        return hpx::parallel::util::loop_idx_n_t<ExPolicy>{}(
            base_idx, it, count, HPX_FORWARD(F, f));
    }

    template <typename ExPolicy, typename Iter, typename CancelToken,
        typename F>
    HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Iter loop_idx_n(
        std::size_t base_idx, Iter it, std::size_t count, CancelToken& tok,
        F&& f)
    {
        return hpx::parallel::util::loop_idx_n_t<ExPolicy>{}(
            base_idx, it, count, tok, HPX_FORWARD(F, f));
    }
#endif

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // Helper class to repeatedly call a function a given number of times
        // starting from a given iterator position.
        template <typename IterCat>
        struct accumulate_n
        {
            template <typename Iter, typename T, typename Pred>
            static constexpr T call(
                Iter it, std::size_t count, T init, Pred&& f)
            {
                for (/**/; count != 0; (void) --count, ++it)
                {
                    init = HPX_INVOKE(f, HPX_MOVE(init), *it);
                }
                return init;
            }
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter, typename T, typename Pred>
    HPX_FORCEINLINE constexpr T accumulate_n(
        Iter it, std::size_t count, T init, Pred&& f)
    {
        using cat = typename std::iterator_traits<Iter>::iterator_category;
        return detail::accumulate_n<cat>::call(
            it, count, HPX_MOVE(init), HPX_FORWARD(Pred, f));
    }

    template <typename T, typename Iter, typename Sent, typename Reduce,
        typename Conv = hpx::identity>
    HPX_FORCEINLINE constexpr T accumulate(
        Iter first, Sent last, Reduce&& r, Conv&& conv = Conv())
    {
        T val = HPX_INVOKE(conv, *first);
        ++first;
        while (first != last)
        {
            val = HPX_INVOKE(r, HPX_MOVE(val), HPX_INVOKE(conv, *first));
            ++first;
        }
        return val;
    }

    template <typename T, typename Iter1, typename Sent1, typename Iter2,
        typename Reduce, typename Conv>
    HPX_FORCEINLINE constexpr T accumulate(
        Iter1 first1, Sent1 last1, Iter2 first2, Reduce&& r, Conv&& conv)
    {
        T val = HPX_INVOKE(conv, *first1, *first2);
        ++first1;
        ++first2;
        while (first1 != last1)
        {
            val = HPX_INVOKE(
                r, HPX_MOVE(val), HPX_INVOKE(conv, *first1, *first2));
            ++first1;
            ++first2;
        }
        return val;
    }
}    // namespace hpx::parallel::util
