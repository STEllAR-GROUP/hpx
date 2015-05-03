//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM)
#define HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/traits/serialize_as_future.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/shared_ptr.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/decay.hpp>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace functional
{
    ///////////////////////////////////////////////////////////////////////////
    struct extract_locality
    {
        typedef naming::id_type result_type;

        extract_locality() {}

        naming::id_type operator()(agas::response const& rep,
            naming::id_type const& id) const
        {
            if (rep.get_status() != success)
            {
                HPX_THROW_EXCEPTION(rep.get_status(),
                    "extract_locality::operator()",
                    boost::str(boost::format(
                        "could not resolve colocated locality for id(%1%)"
                    ) % id));
                return naming::invalid_id;
            }
            return naming::get_id_from_locality_id(rep.get_locality_id());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Bound>
        struct apply_continuation_impl
        {
            typedef typename util::decay<Bound>::type bound_type;

            template <typename T>
            struct result;

            template <typename F, typename T1, typename T2>
            struct result<F(T1, T2)>
              : util::result_of<F(T1, T2)>
            {};

            apply_continuation_impl() {}

            explicit apply_continuation_impl(Bound && bound)
              : bound_(std::move(bound))
            {}

            explicit apply_continuation_impl(
                    Bound && bound, actions::continuation_type const& c)
              : bound_(std::move(bound)),
                cont_(c)
            {}

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t) const
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                if (cont_)
                    bound_.apply_c(cont_, lco, std::forward<T>(t));
                else
                    bound_.apply(lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            BOOST_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar << cont_;
                }
            }

            template <typename Archive>
            BOOST_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar >> cont_;
                }
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            friend struct traits::serialize_as_future<apply_continuation_impl>;

            bound_type bound_;
            actions::continuation_type cont_;
        };
    }

    template <typename Bound>
    functional::detail::apply_continuation_impl<typename util::decay<Bound>::type>
    apply_continuation(Bound && bound)
    {
        return functional::detail::apply_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound));
    }

    template <typename Bound>
    functional::detail::apply_continuation_impl<typename util::decay<Bound>::type>
    apply_continuation(Bound && bound, actions::continuation_type const& c)
    {
        return functional::detail::apply_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound), c);
    }
}}}

namespace hpx { namespace traits
{
    template <typename Bound>
    struct serialize_as_future<
            util::functional::detail::apply_continuation_impl<Bound>
        >
      : traits::serialize_as_future<
            typename util::functional::detail::apply_continuation_impl<
                Bound
            >::bound_type>
    {
        static bool
        call_if(util::functional::detail::apply_continuation_impl<Bound>& b)
        {
            return (b.cont_ && b.cont_->has_to_wait_for_futures()) ||
                traits::serialize_as_future<
                    typename util::functional::detail::apply_continuation_impl<
                        Bound
                    >::bound_type
                >::call_if(b.bound_);
        }

        static void
        call(util::functional::detail::apply_continuation_impl<Bound>& b)
        {
            typedef typename util::functional::detail::apply_continuation_impl<
                    Bound
                >::bound_type bound_type;
            traits::serialize_as_future<bound_type>::call(b.bound_);
            if (b.cont_)
                b.cont_->wait_for_futures();
        }
    };
}}

namespace hpx { namespace util { namespace functional
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Bound>
        struct async_continuation_impl
        {
            typedef typename util::decay<Bound>::type bound_type;

            template <typename T>
            struct result;

            template <typename F, typename T1, typename T2>
            struct result<F(T1, T2)>
              : util::result_of<F(T1, T2)>
            {};

            async_continuation_impl() {}

            explicit async_continuation_impl(Bound && bound)
              : bound_(std::move(bound))
            {}

            explicit async_continuation_impl(
                    Bound && bound, actions::continuation_type const& c)
              : bound_(std::move(bound)),
                cont_(c)
            {}

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t) const
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                if (cont_)
                    bound_.apply_c(cont_, lco, std::forward<T>(t));
                else
                    bound_.apply_c(lco, lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            BOOST_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar << cont_;
                }
            }

            template <typename Archive>
            BOOST_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar >> cont_;
                }
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            friend struct traits::serialize_as_future<async_continuation_impl>;

            bound_type bound_;
            actions::continuation_type cont_;
        };
    }

    template <typename Bound>
    functional::detail::async_continuation_impl<typename util::decay<Bound>::type>
    async_continuation(Bound && bound)
    {
        return functional::detail::async_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound));
    }

    template <typename Bound>
    functional::detail::async_continuation_impl<typename util::decay<Bound>::type>
    async_continuation(Bound && bound, actions::continuation_type const& c)
    {
        return functional::detail::async_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound), c);
    }
}}}

namespace hpx { namespace traits
{
    template <typename Bound>
    struct serialize_as_future<
            util::functional::detail::async_continuation_impl<Bound>
        >
      : traits::serialize_as_future<
            typename util::functional::detail::async_continuation_impl<
                Bound
            >::bound_type>
    {
        static bool
        call_if(util::functional::detail::async_continuation_impl<Bound>& b)
        {
            return (b.cont_ && b.cont_->has_to_wait_for_futures()) ||
                traits::serialize_as_future<
                    typename util::functional::detail::async_continuation_impl<
                        Bound
                    >::bound_type
                >::call_if(b.bound_);
        }

        static void
        call(util::functional::detail::async_continuation_impl<Bound>& b)
        {
            typedef typename util::functional::detail::async_continuation_impl<
                    Bound
                >::bound_type bound_type;
            traits::serialize_as_future<bound_type>::call(b.bound_);
            if (b.cont_)
                b.cont_->wait_for_futures();
        }
    };
}}

#endif
