//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM)
#define HPX_UTIL_DETAIL_COLOCATED_HELPERS_FEB_04_2014_0828PM

#include <hpx/config.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/agas/response.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/unique_ptr.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/decay.hpp>

#include <memory>

#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace functional
{
    ///////////////////////////////////////////////////////////////////////////
    struct extract_locality
    {
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
            HPX_MOVABLE_ONLY(apply_continuation_impl);
        public:
            typedef typename util::decay<Bound>::type bound_type;

            apply_continuation_impl() {}

            explicit apply_continuation_impl(Bound && bound)
              : bound_(std::move(bound))
            {}

            template <typename Continuation>
            explicit apply_continuation_impl(
                    Bound && bound, Continuation && c)
              : bound_(std::move(bound)),
                cont_(new typename
                    util::decay<Continuation>::type(std::forward<Continuation>(c)))
            {}

            apply_continuation_impl(apply_continuation_impl && o)
              : bound_(std::move(o.bound_))
              , cont_(std::move(o.cont_))
            {}

            apply_continuation_impl &operator=(apply_continuation_impl && o)
            {
                bound_ = std::move(o.bound_);
                cont_ = std::move(o.cont_);
                return *this;
            }

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t)
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                if (cont_)
                    bound_.apply_c(std::move(cont_), lco, std::forward<T>(t));
                else
                    bound_.apply(lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            HPX_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar << cont_;
                }
            }

            template <typename Archive>
            HPX_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar >> cont_;
                }
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            bound_type bound_;
            std::unique_ptr<actions::continuation> cont_;
        };
    }

    template <typename Bound>
    functional::detail::apply_continuation_impl<typename util::decay<Bound>::type>
    apply_continuation(Bound && bound)
    {
        return functional::detail::apply_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound));
    }

    template <typename Bound, typename Continuation>
    functional::detail::apply_continuation_impl<typename util::decay<Bound>::type>
    apply_continuation(Bound && bound, Continuation && c)
    {
        return functional::detail::apply_continuation_impl<
            typename util::decay<Bound>::type>(
                std::forward<Bound>(bound), std::forward<Continuation>(c));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename Bound>
        struct async_continuation_impl
        {
            HPX_MOVABLE_ONLY(async_continuation_impl);
        public:
            typedef typename util::decay<Bound>::type bound_type;

            async_continuation_impl() {}

            explicit async_continuation_impl(Bound && bound)
              : bound_(std::move(bound))
            {}

            template <typename Continuation>
            explicit async_continuation_impl(
                    Bound && bound, Continuation && c)
              : bound_(std::move(bound)),
                cont_(new typename util::decay<Continuation>
                        ::type(std::forward<Continuation>(c)))
            {}

            async_continuation_impl(async_continuation_impl && o)
              : bound_(std::move(o.bound_))
              , cont_(std::move(o.cont_))
            {}

            async_continuation_impl &operator=(async_continuation_impl && o)
            {
                bound_ = std::move(o.bound_);
                cont_ = std::move(o.cont_);
                return *this;
            }

            template <typename T>
            typename util::result_of<bound_type(naming::id_type, T)>::type
            operator()(naming::id_type lco, T && t)
            {
                typedef typename util::result_of<
                    bound_type(naming::id_type, T)
                >::type result_type;

                if (cont_)
                    bound_.apply_c(std::move(cont_), lco, std::forward<T>(t));
                else
                    bound_.apply_c(lco, lco, std::forward<T>(t));
                return result_type();
            }

        private:
            // serialization support
            friend class hpx::serialization::access;

            template <typename Archive>
            HPX_FORCEINLINE void save(Archive& ar, unsigned int const) const
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar << cont_;
                }
            }

            template <typename Archive>
            HPX_FORCEINLINE void load(Archive& ar, unsigned int const)
            {
                bool has_continuation = cont_ ? true : false;
                ar & bound_ & has_continuation;
                if (has_continuation)
                {
                    ar >> cont_;
                }
            }

            HPX_SERIALIZATION_SPLIT_MEMBER();

            bound_type bound_;
            std::unique_ptr<actions::continuation> cont_;
        };
    }

    template <typename Bound>
    functional::detail::async_continuation_impl<typename util::decay<Bound>::type>
    async_continuation(Bound && bound)
    {
        return functional::detail::async_continuation_impl<
            typename util::decay<Bound>::type>(std::forward<Bound>(bound));
    }

    template <typename Bound, typename Continuation>
    functional::detail::async_continuation_impl<typename util::decay<Bound>::type>
    async_continuation(Bound && bound, Continuation && c)
    {
        return functional::detail::async_continuation_impl<
            typename util::decay<Bound>::type>(
                std::forward<Bound>(bound), std::forward<Continuation>(c));
    }
}}}

#endif
