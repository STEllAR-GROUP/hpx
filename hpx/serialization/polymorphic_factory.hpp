//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file polymorphic_factory.hpp

#ifndef HPX_SERIALIZATION_POLYMORPHIC_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_FACTORY_HPP

#include <hpx/config.hpp>

#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/detail/count_num_args.hpp>
#include <hpx/traits/needs_automatic_registration.hpp>

#include <map>
#include <string>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace serialization {
    struct input_archive;
    struct output_archive;

    template <typename T>
    struct get_name_impl
    {
        static char const* call()
#ifdef HPX_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
        {
            // If you encounter this assert while compiling code, that means
            // that you have a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION macro
            // somewhere in a source file, but the header in which the function
            // is defined misses a HPX_UTIL_REGISTER_[UNIQUE_]FUNCTION_DECLARATION
            BOOST_STATIC_ASSERT_MSG(
                traits::needs_automatic_registration<T>::value
              , "This type was registered with serialization, but "
                 "HPX_SERIALIZATION_REGISTER_DERIVED_DECLARATION is missing"
            );
            return util::type_id<T>::typeid_.type_id();
        }
#endif
    };

    template <typename T>
    const char * get_name()
    {
        return get_name_impl<T>::call();
    }

    template <typename Base>
    class HPX_EXPORT polymorphic_factory;

    template <typename Base, typename Derived>
    struct polymorphic_factory_registration
    {
        typedef polymorphic_factory<Base> factory_type;
        typedef typename factory_type::pointer_type pointer_type;

        static pointer_type load(input_archive & ar)
        {
            Derived d;
            ar >> d;
            return pointer_type(new Derived(std::move(d)));
        }

        polymorphic_factory_registration()
        {
            factory_type::get_instance().add_factory_function(
                get_name<Derived>()
              , &polymorphic_factory_registration::load
            );
        }
    };

    template <typename Base, typename Derived
      , typename Enable =
            typename traits::needs_automatic_registration<Derived>::type
    >
    struct polymorphic_factory_auto_registration
    {
        polymorphic_factory_auto_registration()
        {
            polymorphic_factory_registration<Base, Derived> auto_register;
        }

        polymorphic_factory_auto_registration & register_function()
        {
            return *this;
        }
    };

    template <typename Base, typename Derived>
    struct polymorphic_factory_auto_registration<Base, Derived>
    {
        polymorphic_factory_auto_registration()
        {
        }

        polymorphic_factory_auto_registration & register_function()
        {
        }
    };

    template <typename Base>
    class HPX_EXPORT polymorphic_factory
    {
    public:
        typedef HPX_STD_UNIQUE_PTR<Base> pointer_type;
        typedef pointer_type(*loader_type)(input_archive & ar);
        typedef std::multimap<
            boost::uint32_t, std::pair<std::string, loader_type>
        > loader_map;

        template <typename Archive>
        static pointer_type load(Archive & ar, std::string const & name)
        {
            polymorphic_factory<Base> const & factory = get_instance();
            typename loader_map::const_iterator it = factory.locate(
                util::jenkins_hash()(name), name);

            if (it != factory.loader_map_.end())
                return ((*it).second.second)(ar);

            std::string error = "Can not find '";
            error += name;
            error += "' in type registry";
            HPX_ASSERT(false);
/*
        HPX_THROW_EXCEPTION(bad_action_code
            , "polymorphic_factory"BOOST_PP_STRINGIZE(Name)"::create"
            , error);
*/
            return pointer_type();
        }

    private:
        static polymorphic_factory& get_instance();

        typename loader_map::const_iterator locate(
            boost::uint32_t hash
          , std::string const & name) const
        {
            typedef std::pair<
                typename loader_map::const_iterator, typename loader_map::const_iterator
            > equal_range_type;

            equal_range_type r = loader_map_.equal_range(hash);
            if (r.first != r.second)
            {
                typename loader_map::const_iterator it = r.first;
                if (++it == r.second)
                {
                    // there is only one match in the map
                    return r.first;
                }

                // there is more than one entry with the same hash in the map
                for (it = r.first; it != r.second; ++it)
                {
                    if ((*it).second.first == name)
                        return it;
                }

                // fall through...
            }
            return loader_map_.end();
        }

        void add_factory_function(
            std::string const & name
          , loader_type loader)
        {
            boost::uint32_t hash = util::jenkins_hash()(name);
            typename loader_map::const_iterator it = locate(hash, name);
            if (it != loader_map_.end())
                return;

            loader_map_.insert(std::make_pair(hash, std::make_pair(name, loader)));
        }

        loader_map loader_map_;

        template <typename B, typename D>
        friend struct polymorphic_factory_registration;
    };
}}

#define HPX_SERIALIZATION_REGISTER_BASE(BaseType)                               \
    namespace hpx { namespace serialization {                                   \
        template <typename Base>                                                \
        polymorphic_factory<Base> & polymorphic_factory<Base>::get_instance()   \
        {                                                                       \
            util::static_<polymorphic_factory<Base> > factory;                  \
            return factory.get();                                               \
        }                                                                       \
                                                                                \
        template class polymorphic_factory<BaseType>;                           \
    }}                                                                          \
/**/

#define HPX_SERIALIZATION_REGISTER_DERIVED_3(Base, Derived, Name)               \
    namespace hpx { namespace serialization {                                   \
        template<> HPX_ALWAYS_EXPORT                                            \
        char const* get_name<Derived>()                                         \
        {                                                                       \
            return BOOST_PP_STRINGIZE(Name);                                    \
        }                                                                       \
    }}                                                                          \
    static hpx::serialization::polymorphic_factory_registration<                \
        Base                                                                    \
      , Derived                                                                 \
    > const BOOST_PP_CAT(Name, _polymorphic_factory_registration) =             \
        hpx::serialization::polymorphic_factory_registration<                   \
            Base                                                                \
          , Derived                                                             \
        >();                                                                    \
/**/

#define HPX_SERIALIZATION_REGISTER_DERIVED_2(Base, Derived)                     \
    HPX_SERIALIZATION_REGISTER_DERIVED_3(Base, Derived, Derived)                \
/**/

#define HPX_SERIALIZATION_REGISTER_DERIVED_(...)                                \
    HPX_UTIL_EXPAND_(BOOST_PP_CAT(                                              \
        HPX_SERIALIZATION_REGISTER_DERIVED_, HPX_UTIL_PP_NARG(__VA_ARGS__)      \
    )(__VA_ARGS__))                                                             \
/**/

#define HPX_SERIALIZATION_REGISTER_DERIVED(...)                                 \
    HPX_SERIALIZATION_REGISTER_DERIVED_(__VA_ARGS__)                            \
/**/

#define HPX_SERIALIZATION_REGISTER_DERIVED_DECLARATION(Derived)                 \
    namespace hpx { namespace serialization {                                   \
        template <> HPX_ALWAYS_EXPORT                                           \
        const char * get_name<Derived>();                                       \
    }}                                                                          \
    namespace hpx { namespace traits {                                          \
        template <>                                                             \
        struct needs_automatic_registration<Derived>                            \
          : boost::mpl::false_                                                  \
        {};                                                                     \
    }}                                                                          \
/**/

#endif
