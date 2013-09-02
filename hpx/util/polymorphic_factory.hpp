//  Copyright (c) 2013 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file polymorphic_factory.hpp

#if !defined(HPX_RUNTIME_UTIL_POLYMORPHIC_FACTORY_HPP)
#define HPX_RUNTIME_UTIL_POLYMORPHIC_FACTORY_HPP

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/static.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/cstdint.hpp>

#include <map>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace traits
{
    // This trait is used to decide whether a class (or specialization) is
    // required to automatically register to the action factory
    template <typename T, typename Enable = void>
    struct needs_automatic_registration
      : boost::mpl::true_
    {};
}}

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    struct action_registration;

    template <typename Continuation>
    struct continuation_registration;
}}}

namespace hpx { namespace util
{
    namespace detail
    {
        template <typename Function>
        struct function_registration;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Base>
    class HPX_EXPORT polymorphic_factory
    {
    public:
        typedef boost::shared_ptr<Base>(*ctor_type)();
        typedef std::multimap<
            boost::uint32_t, std::pair<std::string, ctor_type>
        > ctor_map;

        static boost::shared_ptr<Base> create(std::string const & name);

    protected:
        typename ctor_map::const_iterator locate(boost::uint32_t hash,
            std::string const& name) const;

    private:
        void add_factory_function(std::string const & name, ctor_type ctor);
        static polymorphic_factory& get_instance();

        ctor_map ctor_map_;

        template <typename Action>
        friend struct actions::detail::action_registration;

        template <typename Continuation>
        friend struct actions::detail::continuation_registration;

        template <typename Function>
        friend struct util::detail::function_registration;
    };
}}

#include <hpx/config/warnings_suffix.hpp>

#endif

