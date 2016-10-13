//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_ACTIONS_CONTINUATION_IMPL_HPP
#define HPX_ACTIONS_CONTINUATION_IMPL_HPP

#include <hpx/runtime/applier/detail/apply_implementations_fwd.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/serialization/access.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>

#include <utility>

namespace hpx { namespace actions {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Cont>
    struct continuation_impl
    {
    private:
        typedef typename util::decay<Cont>::type cont_type;

    public:
        continuation_impl() {}

        template <typename Cont_>
        continuation_impl(Cont_ && cont, hpx::naming::id_type const& target)
          : cont_(std::forward<Cont_>(cont)), target_(target)
        {}

        virtual ~continuation_impl() {}

        template <typename T>
        typename util::result_of<cont_type(hpx::naming::id_type, T)>::type
        operator()(hpx::naming::id_type const& lco, T && t) const
        {
            hpx::apply_c(cont_, lco, target_, std::forward<T>(t));

            // Unfortunately we need to default construct the return value,
            // this possibly imposes an additional restriction of return types.
            typedef typename util::result_of<
                cont_type(hpx::naming::id_type, T)
            >::type result_type;
            return result_type();
        }

    private:
        // serialization support
        friend class hpx::serialization::access;

        template <typename Archive>
        HPX_FORCEINLINE void serialize(Archive& ar, unsigned int const)
        {
            ar & cont_ & target_;
        }

        cont_type cont_;
        hpx::naming::id_type target_;
    };
}}

#endif
