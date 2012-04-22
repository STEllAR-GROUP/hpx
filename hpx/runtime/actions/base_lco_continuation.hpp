//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_BASE_LCO_CONTINUATION_APR_14_2012_0559PM)
#define HPX_RUNTIME_ACTIONS_BASE_LCO_CONTINUATION_APR_14_2012_0559PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    template <typename Action, typename Arg0>
    inline bool
    apply(naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0);

    namespace actions
    {
        ///////////////////////////////////////////////////////////////////////
        // A base_lco_continuation is a special continuation which simply sends
        // the result value back to the destination as specified by the target
        // id_type (stored in the continuation base class).
        template <typename Result>
        struct base_lco_continuation : typed_continuation<Result>
        {
            base_lco_continuation()
            {}

            explicit base_lco_continuation(naming::id_type const& gid)
              : typed_continuation<Result>(gid)
            {}

            ~base_lco_continuation ()
            {
                detail::guid_initialization<base_lco_continuation>();
            }

            void trigger_value(BOOST_RV_REF(Result) result) const
            {
                typedef typename
                    lcos::template base_lco_with_value<Result>::set_value_action
                set_value_action_type;

                LLCO_(info) << "continuation::trigger(" << this->get_gid() << ")";

                hpx::apply<set_value_action_type>(this->get_gid(), boost::move(result));
            }

            static void register_base()
            {
                util::void_cast_register_nonvirt<
                    base_lco_continuation, base_type>();
                base_type::register_base();
            }

        private:
            /// serialization support
            friend class boost::serialization::access;
            typedef typed_continuation<Result> base_type;

            template <class Archive>
            void serialize(Archive& ar, const unsigned int /*version*/)
            {
                ar & util::base_object_nonvirt<base_type>(*this);
            }

            static detail::register_base_helper<base_lco_continuation> helper_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        detail::register_base_helper<base_lco_continuation<Result> >
            base_lco_continuation<Result>::helper_ =
                detail::register_base_helper<base_lco_continuation<Result> >();
    }
}

///////////////////////////////////////////////////////////////////////////////
// registration code for serialization
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result>),
    (hpx::actions::base_lco_continuation<Result>)
)

#include <hpx/config/warnings_suffix.hpp>

#endif
