//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_SEGMENTED_ALGORITHM_ALL_ANY_NONE_AUG_12_2017_0836PM)
#define HPX_PARALLEL_SEGMENTED_ALGORITHM_FOR_EACH_OCT_15_2014_0839PM

#include <hpx/config.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>

#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/all_any_none.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/segmented_algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_remote_exceptions.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <list>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1
{
    ///////////////////////////////////////////////////////////////////////////
    // segmented_all_any_none
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        /// \cond NOINTERNAL

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_none_of(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, F && f, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            bool output = false;
            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }

                // handle all of the full partitions
                for (++sit; sit != send && output; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        output = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end && output)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }
            }

            return result::get(std::move(output));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_none_of(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, F && f, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<bool> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, f
                        ));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<shared_future<bool> > && r)
                        ->  bool
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);
                        std::vector<bool> res =
                            hpx::util::unwrap(std::move(r));
                        auto it = res.begin();
                        while (it != res.end())
                        {
                            if(*it == false)
                                return false;
                            it++;
                        }
                        return true;
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        none_of_(ExPolicy && policy, FwdIter first, FwdIter last,
            F && f, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;

            return segmented_none_of(
                none_of(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), is_seq());
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        none_of_(ExPolicy && policy, FwdIter first, FwdIter last,
            F && f, std::false_type);

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_any_of(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, F && f, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            bool output = false;
            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }

                // handle all of the full partitions
                for (++sit; sit != send && !output; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        output = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end && !output)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }
            }

            return result::get(std::move(output));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_any_of(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, F && f, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<bool> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, f
                        ));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<shared_future<bool> > && r)
                        ->  bool
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);
                        std::vector<bool> res =
                            hpx::util::unwrap(std::move(r));
                        auto it = res.begin();
                        while (it != res.end())
                        {
                            if(*it == true)
                                return true;
                            it++;
                        }
                        return false;
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        any_of_(ExPolicy && policy, FwdIter first, FwdIter last,
            F && f, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;

            return segmented_any_of(
                any_of(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), is_seq());
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        any_of_(ExPolicy && policy, FwdIter first, FwdIter last,
            F && f, std::false_type);

        // sequential remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_all_of(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, F && f, std::true_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);
            bool output = false;
            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);

                if (beg != end)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }

                // handle all of the full partitions
                for (++sit; sit != send && output; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);

                    if (beg != end)
                    {
                        output = dispatch(traits::get_id(sit),
                            algo, policy, std::true_type(), beg, end, f
                        );
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end && output)
                {
                    output = dispatch(traits::get_id(sit),
                        algo, policy, std::true_type(), beg, end, f
                    );
                }
            }

            return result::get(std::move(output));
        }

        // parallel remote implementation
        template <typename Algo, typename ExPolicy, typename FwdIter,
            typename F>
        static typename util::detail::algorithm_result<ExPolicy, bool>::type
        segmented_all_of(Algo && algo, ExPolicy const& policy,
            FwdIter first, FwdIter last, F && f, std::false_type)
        {
            typedef hpx::traits::segmented_iterator_traits<FwdIter> traits;
            typedef typename traits::segment_iterator segment_iterator;
            typedef typename traits::local_iterator local_iterator_type;
            typedef util::detail::algorithm_result<ExPolicy, bool> result;

            typedef std::integral_constant<bool,
                    !hpx::traits::is_forward_iterator<FwdIter>::value
                > forced_seq;

            segment_iterator sit = traits::segment(first);
            segment_iterator send = traits::segment(last);

            std::vector<shared_future<bool> > segments;
            segments.reserve(std::distance(sit, send));

            if (sit == send)
            {
                // all elements are on the same partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }
            }
            else {
                // handle the remaining part of the first partition
                local_iterator_type beg = traits::local(first);
                local_iterator_type end = traits::end(sit);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }

                // handle all of the full partitions
                for (++sit; sit != send; ++sit)
                {
                    beg = traits::begin(sit);
                    end = traits::end(sit);
                    if (beg != end)
                    {
                        segments.push_back(dispatch_async(traits::get_id(sit),
                            algo, policy, forced_seq(), beg, end, f
                        ));
                    }
                }

                // handle the beginning of the last partition
                beg = traits::begin(sit);
                end = traits::local(last);
                if (beg != end)
                {
                    segments.push_back(dispatch_async(traits::get_id(sit),
                        algo, policy, forced_seq(), beg, end, f
                    ));
                }
            }

            return result::get(
                dataflow(
                    [=](std::vector<shared_future<bool> > && r)
                        ->  bool
                    {
                        // handle any remote exceptions, will throw on error
                        std::list<std::exception_ptr> errors;
                        parallel::util::detail::handle_remote_exceptions<
                            ExPolicy
                        >::call(r, errors);
                        std::vector<bool> res =
                            hpx::util::unwrap(std::move(r));
                        auto it = res.begin();
                        while (it != res.end())
                        {
                            if(*it == false)
                                return false;
                            it++;
                        }
                        return true;
                    },
                    std::move(segments)));
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        all_of_(ExPolicy && policy, FwdIter first, FwdIter last,
            F && f, std::true_type)
        {
            typedef parallel::execution::is_sequenced_execution_policy<
                    ExPolicy
                > is_seq;

            typedef hpx::traits::segmented_iterator_traits<FwdIter>
                iterator_traits;

            return segmented_all_of(
                all_of(),
                std::forward<ExPolicy>(policy), first, last,
                std::forward<F>(f), is_seq());
        }

        template <typename ExPolicy, typename FwdIter, typename F>
        typename util::detail::algorithm_result<ExPolicy, bool>::type
        all_of_(ExPolicy && policy, FwdIter first, FwdIter last,
            F && f, std::false_type);

      /// \endcond
    }
}}}

#endif
