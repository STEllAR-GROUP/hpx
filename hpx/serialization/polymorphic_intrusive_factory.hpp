//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP

#include <hpx/config.hpp>
#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/static.hpp>

#include <boost/noncopyable.hpp>
#include <boost/unordered_map.hpp>
#include <boost/mpl/bool.hpp>

namespace hpx { namespace serialization {

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
      map_[name] = fun;
    }

    template <class T>
    T* create(const std::string& name) const
    {
      return static_cast<T*>(map_.at(name)());
    }

  private:
    polymorphic_intrusive_factory()
    {
    }

    friend struct hpx::util::static_<polymorphic_intrusive_factory>;

    ctor_map_type map_;
  };

}}

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name)        \
  virtual const char* hpx_serialization_get_name() const                      \
  {                                                                           \
    return Name;                                                              \
  }                                                                           \
  static void* factory_function()                                             \
  {                                                                           \
    return new Class;                                                         \
  }                                                                           \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, Name)                  \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    serialize<hpx::serialization::input_archive>(ar, n);                      \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    static bool register_class = (                                            \
      hpx::serialization::polymorphic_intrusive_factory::instance().          \
        register_class(                                                       \
          Class::hpx_serialization_get_name(),                                \
          &Class::factory_function                                            \
        ),                                                                    \
      true                                                                    \
    );                                                                        \
    const_cast<Class*>(this)->                                                \
      serialize<hpx::serialization::output_archive>(ar, n);                   \
  }                                                                           \
  HPX_SERIALIZATION_SPLIT_MEMBER();                                           \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);             \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(Class, Name)         \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    load<hpx::serialization::input_archive>(ar, n);                           \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    static bool register_class = (                                            \
      hpx::serialization::polymorphic_intrusive_factory::instance().          \
        register_class(                                                       \
          Class::hpx_serialization_get_name(),                                \
          &Class::factory_function                                            \
        ),                                                                    \
      true                                                                    \
    );                                                                        \
    save<hpx::serialization::output_archive>(ar, n);                          \
  }                                                                           \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);             \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(Class)                         \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    serialize<hpx::serialization::input_archive>(ar, n);                      \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    const_cast<Class*>(this)->                                                \
      serialize<hpx::serialization::output_archive>(ar, n);                   \
  }                                                                           \
  HPX_SERIALIZATION_SPLIT_MEMBER()                                            \
  virtual const char*                                                         \
    hpx_serialization_get_name() const = 0;                                   \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT_SPLITTED(Class)                \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    load<hpx::serialization::input_archive>(ar, n);                           \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    save<hpx::serialization::output_archive>(ar, n);                          \
  }                                                                           \
  virtual const char*                                                         \
    hpx_serialization_get_name() const = 0;                                   \
/**/

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class)                        \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(                          \
      Class, BOOST_STRINGIZE(Class))                                          \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC(Class)                                  \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, BOOST_STRINGIZE(Class))      \

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED(Class)                         \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                           \
      Class, BOOST_STRINGIZE(Class))                                          \
/**/

#endif
