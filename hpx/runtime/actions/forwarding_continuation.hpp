//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_RUNTIME_ACTIONS_FORWARDING_CONTINUATION_APR_14_2012_0613PM)
#define HPX_RUNTIME_ACTIONS_FORWARDING_CONTINUATION_APR_14_2012_0613PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/detail/serialization_registration.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace actions
{
    ///////////////////////////////////////////////////////////////////////////
    // A forwarding_continuation is a special continuation which invokes the
    // stored function object to trigger the continuation.
    //
    // The function object F is assumed to have the prototype
    //
    //      void F (id_type cont, Result);
    //
    // Result is the return type of the action this forwarding_continuation
    // is associated with. The function object is assumed to invoke a
    // follow-on action which receives the result. The id references the
    // object where the overall result has to be sent to.
    template <typename Result, typename F>
    struct forwarding_continuation : typed_continuation<Result>
    {
        forwarding_continuation()
        {}

        explicit forwarding_continuation(naming::id_type const& id,
                BOOST_RV_REF(F) f)
          : typed_continuation<Result>(id), f_(boost::move(f))
        {}

        ~forwarding_continuation ()
        {
            detail::guid_initialization<forwarding_continuation>();
        }

        void trigger(BOOST_RV_REF(Result) result) const
        {
            LLCO_(info) << "forwarding_continuation::trigger("
                        << this->get_gid() << ")";

            f_(this->get_gid(), boost::move(result));
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

        static void register_base()
        {
            util::void_cast_register_nonvirt<
                forwarding_continuation, base_type>();
            base_type::register_base();
        }

        static detail::register_base_helper<forwarding_continuation> helper_;

        F f_;
    };

    ///////////////////////////////////////////////////////////////////////
    template <typename Result, typename F>
    detail::register_base_helper<forwarding_continuation<Result, F> >
        forwarding_continuation<Result, F>::helper_ =
            detail::register_base_helper<forwarding_continuation<Result, F> >();

    ///////////////////////////////////////////////////////////////////////
    template <typename Result, typename F>
    forwarding_continuation<Result, F>*
    create_forwarding_continuation(naming::id_type const& id,
        BOOST_FWD_REF(F) f)
    {
        return new forwarding_continuation<Result, F>(id, f);
    }
}}

///////////////////////////////////////////////////////////////////////////////
// registration code for serialization
HPX_SERIALIZATION_REGISTER_TEMPLATE(
    (template <typename Result>),
    (hpx::actions::forwarding_continuation<Result>)
)

#include <hpx/config/warnings_suffix.hpp>

#endif
