//  Copyright (c) 2007-2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM)
#define HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/agas/response.hpp>
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

        extract_locality(naming::id_type const& id) : id_(id) {}

        naming::id_type operator()(agas::response const& rep) const
        {
            if (rep.get_status() != success)
            {
                HPX_THROW_EXCEPTION(rep.get_status(),
                    "extract_locality::operator()",
                    boost::str(boost::format(
                        "could not resolve colocated locality for id(%1%)"
                    ) % id_));
                return naming::invalid_id;
            }
            return naming::get_id_from_locality_id(rep.get_locality_id());
        }

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const version)
        {
            ar & id_;
        }

        naming::id_type id_;
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

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t) const
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                bound_.apply(lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class boost::serialization::access;

            template <typename Archive>
            BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
            {
                ar & bound_;
            }

            bound_type bound_;
        };
    }

    template <typename Bound>
    functional::detail::apply_continuation_impl<typename util::decay<Bound>::type>
    apply_continuation(Bound && bound)
    {
        return functional::detail::apply_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound));
    }

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

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t) const
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                bound_.apply_c(lco, lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class boost::serialization::access;

            template <typename Archive>
            BOOST_FORCEINLINE void serialize(Archive& ar, unsigned int const)
            {
                ar & bound_;
            }

            bound_type bound_;
        };
    }

    template <typename Bound>
    functional::detail::async_continuation_impl<typename util::decay<Bound>::type>
    async_continuation(Bound && bound)
    {
        return functional::detail::async_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound));
    }
}}}

#endif
