//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_ACTIONS_INVOCATION_COUNT_REGISTRY_SEP_25_2015_0727AM)
#define HPX_ACTIONS_INVOCATION_COUNT_REGISTRY_SEP_25_2015_0727AM

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/performance_counters/counters.hpp>

#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/safe_lexical_cast.hpp>
#include <hpx/util/static.hpp>

#include <boost/preprocessor/stringize.hpp>
#include <boost/noncopyable.hpp>
#include <boost/unordered_map.hpp>
#include <boost/atomic.hpp>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace actions { namespace detail
{
    class HPX_EXPORT invocation_count_registry : boost::noncopyable
    {
    public:
        typedef boost::int64_t (*get_invocation_count_type)(bool);
        typedef boost::unordered_map<
                std::string, get_invocation_count_type, hpx::util::jenkins_hash
            > map_type;

        static invocation_count_registry& instance();

        void register_class(std::string const& name, get_invocation_count_type fun);

        get_invocation_count_type
            get_invocation_counter(std::string const& name) const;

        bool counter_discoverer(
            performance_counters::counter_info const& info,
            performance_counters::counter_path_elements const& p,
            performance_counters::discover_counter_func const& f,
            performance_counters::discover_counters_mode mode, error_code& ec);

    private:
        friend struct hpx::util::static_<invocation_count_registry>;

        map_type map_;
    };

//     template <typename T, typename = void>
//     struct register_class_name
//     {
//         register_class_name()
//         {
//             invocation_count_registry::instance().
//                 register_class(
//                     T::hpx_serialization_get_name_impl(),
//                     &factory_function
//                 );
//         }
//
//         static void* factory_function()
//         {
//             return new T;
//         }
//
//         register_class_name& instantiate()
//         {
//             return *this;
//         }
//
//         static register_class_name instance;
//     };
//
//     template <class T, class Enable>
//     register_class_name<T, Enable> register_class_name<T, Enable>::instance;

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
