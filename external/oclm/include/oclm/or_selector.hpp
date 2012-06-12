
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_OR_SELECTOR_HPP
#define OCLM_OR_SELECTOR_HPP

#include <oclm/selector.hpp>

namespace oclm
{
    namespace detail
    {
        template <typename S1, typename S2>
        struct or_selector
        {
            S1 s1;
            S2 s2;

            bool
            operator()(platform const & t, std::vector<platform> &ts) const
            {
                return
                    invoke(s1, t, ts, typename is_platform_selector<S1>::type())
                 || invoke(s2, t, ts, typename is_platform_selector<S2>::type());
            }

            bool
            operator()(device const & t, std::vector<device> &ts) const
            {
                return
                    invoke(s1, t, ts, typename is_device_selector<S1>::type())
                 || invoke(s2, t, ts, typename is_device_selector<S2>::type());
            }

            template <typename S, typename T>
            bool
            invoke(S const & s, T const & p, std::vector<T> & ps, boost::mpl::true_) const
            {
                return s(p, ps);
            }

            template <typename S, typename T>
            bool
            invoke(S const &, T const &, std::vector<T> &, boost::mpl::false_) const
            {
                return false;
            }
        };
    }

    template <typename S1, typename S2>
    struct is_platform_selector<detail::or_selector<S1, S2> >
        : boost::mpl::or_<
            typename is_platform_selector<S1>::type
          , typename is_platform_selector<S2>::type
        >
    {};

    template <typename S1, typename S2>
    struct is_device_selector<detail::or_selector<S1, S2> >
        : boost::mpl::or_<
            typename is_device_selector<S1>::type
          , typename is_device_selector<S2>::type
        >
    {};

    template <typename F1, typename F2>
    typename boost::enable_if<
        boost::mpl::and_<
            typename is_selector<F1>::type
          , typename is_selector<F2>::type
        >
      , selector<detail::or_selector<F1, F2> >
    >::type
    operator||(F1 const & s1, F2 const & s2)
    {
        detail::or_selector<F1, F2 > sel = {s1, s2};
        return selector<detail::or_selector<F1, F2> >(sel);
    }
}

#endif
