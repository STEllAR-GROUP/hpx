//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/util/demangle_helper.hpp>
#include <hpx/util/detail/pp/stringize.hpp>
#include <hpx/util/jenkins_hash.hpp>

#include <string>
#include <unordered_map>

namespace hpx { namespace serialization { namespace detail
{
    class polymorphic_intrusive_factory
    {
    public:
        HPX_NON_COPYABLE(polymorphic_intrusive_factory);

    private:
        typedef void* (*ctor_type) ();
        typedef std::unordered_map<std::string,
            ctor_type, hpx::util::jenkins_hash> ctor_map_type;

    public:
        polymorphic_intrusive_factory() {}

        HPX_EXPORT static polymorphic_intrusive_factory& instance();

        HPX_EXPORT void register_class(std::string const& name, ctor_type fun);

        HPX_EXPORT void* create(std::string const& name) const;

        template <typename T>
        T* create(std::string const& name) const
        {
            return static_cast<T*>(create(name));
        }

    private:
        ctor_map_type map_;
    };

    template <typename T, typename Enable = void>
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

    template <typename T, typename Enable>
    register_class_name<T, Enable> register_class_name<T, Enable>::instance;

}}}

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name)        \
  template <typename, typename> friend                                        \
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
      return Class::hpx_serialization_get_name_impl();                        \
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
  HPX_SERIALIZATION_SPLIT_MEMBER()                                            \
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
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, HPX_PP_STRINGIZE(Class))     \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED(Class)                         \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                           \
      Class, HPX_PP_STRINGIZE(Class))                                         \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE(Class)                         \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(                                    \
      Class, hpx::util::type_id<Class>::typeid_.type_id();)                   \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED(Class)                \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                           \
      Class, hpx::util::type_id<T>::typeid_.type_id();)                       \
/**/

#endif
