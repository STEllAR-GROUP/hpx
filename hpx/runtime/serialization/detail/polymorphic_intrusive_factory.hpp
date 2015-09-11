//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP

#include <hpx/exception.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/util/static.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/noncopyable.hpp>
#include <boost/unordered_map.hpp>
#include <boost/atomic.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace serialization { namespace detail
{
    class HPX_EXPORT polymorphic_intrusive_factory: boost::noncopyable
    {
    public:
        typedef void* (*ctor_type) ();
        typedef boost::unordered_map<std::string,
                ctor_type, hpx::util::jenkins_hash> ctor_map_type;

        static polymorphic_intrusive_factory& instance()
        {
            hpx::util::static_<polymorphic_intrusive_factory> factory;
            return factory.get();
        }

        void register_class(const std::string& name, ctor_type fun)
        {
            if(name.empty())
            {
                HPX_THROW_EXCEPTION(serialization_error
                  , "polymorphic_intrusive_factory::register_class"
                  , "Cannot register a factory with an empty name");
            }
            auto it = map_.find(name);
            if(it == map_.end())
            {
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
                map_.emplace(name, fun);
#else
                map_.insert(ctor_map_type::value_type(name, fun));
#endif
            }
        }

        template <class T>
        T* create(const std::string& name) const
        {
            return static_cast<T*>(map_.at(name)());
        }

        friend struct hpx::util::static_<polymorphic_intrusive_factory>;

        ctor_map_type map_;
    };

    template <class T, typename = void>
    struct register_class_name
    {
        register_class_name()
        {
            polymorphic_intrusive_factory::instance().
              register_class(
                T::hpx_serialization_get_name_impl(),
                &factory_function
              );
        }

        static void* factory_function()
        {
            return new T;
        }

        register_class_name& instantiate()
        {
            return *this;
        }

        static register_class_name instance;
    };

    template <class T, class Enable>
    register_class_name<T, Enable> register_class_name<T, Enable>::instance;

}}}

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name)        \
  template <class, class> friend                                              \
  struct ::hpx::serialization::detail::register_class_name;                   \
                                                                              \
  static std::string hpx_serialization_get_name_impl()                        \
  {                                                                           \
      hpx::serialization::detail::register_class_name<                        \
          Class>::instance.instantiate();                                     \
      return Name;                                                            \
  }                                                                           \
  virtual std::string hpx_serialization_get_name() const                      \
  {                                                                           \
      return Class ::hpx_serialization_get_name_impl();                       \
  }                                                                           \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, Name)                  \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);             \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
      serialize<hpx::serialization::input_archive>(ar, n);                    \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
      const_cast<Class*>(this)->                                              \
          serialize<hpx::serialization::output_archive>(ar, n);               \
  }                                                                           \
  HPX_SERIALIZATION_SPLIT_MEMBER();                                           \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(Class, Name)         \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);             \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
      load<hpx::serialization::input_archive>(ar, n);                         \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
      save<hpx::serialization::output_archive>(ar, n);                        \
  }                                                                           \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(Class)                         \
  virtual std::string hpx_serialization_get_name() const = 0;                 \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
      serialize<hpx::serialization::input_archive>(ar, n);                    \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
      const_cast<Class*>(this)->                                              \
          serialize<hpx::serialization::output_archive>(ar, n);               \
  }                                                                           \
  HPX_SERIALIZATION_SPLIT_MEMBER()                                            \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT_SPLITTED(Class)                \
  virtual std::string hpx_serialization_get_name() const = 0;                 \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
      load<hpx::serialization::input_archive>(ar, n);                         \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
      save<hpx::serialization::output_archive>(ar, n);                        \
  }                                                                           \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC(Class)                                  \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, BOOST_PP_STRINGIZE(Class))   \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED(Class)                         \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                           \
      Class, BOOST_PP_STRINGIZE(Class))                                       \
/**/

#include <hpx/config/warnings_suffix.hpp>

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE(Class)                         \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(                                    \
      Class, hpx::util::type_id<Class>::typeid_.type_id();)                   \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED(Class)                \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                           \
      Class, hpx::util::type_id<T>::typeid_.type_id();)                       \
/**/

#endif
