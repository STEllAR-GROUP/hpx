#include <hpx/include/async.hpp>
#include <hpx/include/dataflow.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/parallel_set_operations.hpp>
#include <hpx/include/parallel_is_sorted.hpp>
#include <hpx/include/parallel_sort.hpp>
#include <hpx/include/parallel_find.hpp>

#include <boost/atomic.hpp>

#include <vector>
#include <initializer_list>

namespace domain_map
{
  namespace detail
  {
    template<typename T>
    struct local_domain_descriptor : hpx::components::simple_component_base<local_domain_descriptor<T> >
    {
      // local index_sets on each locality
      std::vector<T> local_index_set;
      std::size_t    index_size;

      //  I think it could be good to have map like this
      //  local_index_set  < ---- > data
      //
      public:

             local_domain_descriptor() {} noexcept;
             // functions
              /*
                query which locality
              */
             local_domain_descriptor(const std::initializer_list<T>& to_assign)
                      :
                        local_index_set(to_assign) ,
                        index_size(to_assign.size() {}
            /*  local_domain_descriptor(std::initializer_list<T> && to_assign)
                      :
                        local_index_set(to_assign) ,
                        index_size(to_assign.size()) {}
            */
            local_domain_descriptor(const std::vector<T>& to_assign)
                      :
                        local_index_set(to_assign) ,
                        index_size(to_assign.size()) {}
            /////////////////////////
            // accessors
            bool fetch_indexes(T const& index_)
              {
                local_index_set.emplace_back(index);
                index_size ++;
                return true;
              }
            bool remove_index(const T & index_)
              {
                using namespace hpx::parallel;
                auto fut_remove_index = hpx::parallel::find(par(task),
                local_index_set.begin(),local_index_set.end(),index_);

                auto remove_index = fut_remove_index.get();
                if( remove_index != local_index_set.end())
                  {
                    local_index_set.erase(remove_index);
                    index_size --;
                    return true;
                  }
                  return false;
              }

              bool remove_index(T && index_)
                {
                  using namespace hpx::parallel;
                  auto fut_remove_index = hpx::parallel::find(par(task),
                  local_index_set.begin(),local_index_set.end(),std::move(index_));

                  auto remove_index = fut_remove_index.get();
                  if( remove_index != local_index_set.end())
                    {
                      local_index_set.erase(remove_index);
                      index_size --;
                      return true;
                    }
                    return false;
                }
              bool add_index(const T& value)
              {
                local_index_set.emplace_back(value);
                return true;
              }

              bool add_index(T&& value)
              {
                local_index_set.emplace_back(std::move(value));
                return true;
              }
            HPX_DEFINE_COMPONENT_ACTION(local_domain_descriptor, fetch_indexes);
            HPX_DEFINE_COMPONENT_ACTION(local_domain_descriptor, remove_index);
            HPX_DEFINE_COMPONENT_ACTION(local_domain_descriptor, add_index);
    };


    template<>
    struct local_domain_descriptor<arithmetic> : hpx::components::simple_component_base<
                                                  local_domain_descriptor<arithmetic> >
    {

    };
  }
  using namespace detail;
  using base_type = hpx::components::simple_component_base<local_domain_descriptor<T> >
  HPX_REGISTER_COMPONENT(base_type, local_domain_descriptor);
  HPX_REGISTER_ACTION(base_type::fetch_indexes);
  HPX_REGISTER_ACTION(base_type::remove_index);
  HPX_REGISTER_ACTION(base_type::add_index);


    template<typename T>
    struct global_domain_map : simple_component_base<global_domain_map<T> >
      {
        // Non-specialization of non-arithmetic domains
        public:
               typedef T          value_type;
               typedef T*         pointer;
               typedef const T*   const pointer;
               typedef T&         reference;
               typedef const T&   const_reference;
               typedef unsigned int size_type;

              // iterators stuff's
        public:
                // Constructor
                global_domain_map() : localities_(hpx::find_here()) {}

                global_domain_map(std::initializer_list<T>& tem ) :
                                               index_set(tem) ,
                                               index_size(tem.size()),
                                               localities_(hpx::find_here()) {}

                 global_domain_map(std::initializer_list<T>&& tem) :
                                               index_set(std::move(tem)),
                                               index_size(tem.size()),
                                               localities_(hpx::find_here())    {}

                 global_domain_map(const global_domain_map& other) :
                             localities_(other.localities_),
                              index_set(other.index_set),
                              index_size(other.index_size) {}

                 global_domain_map(global_domain_map&& other) :
                                          localities_(std::move(other.localities_)),
                                           index_set(std::move(other.index_set)),
                                           index_size(std::move(other.index_size)) {}

                  ~global_domain_map();

                  global_domain_map& operator= (const global_domain_map& other)
                  {
                      localities_  = other.localities_;
                      index_set    = other.index_set;
                      localities_  = other.localities_;
                      return *this;
                  }

                  global_domain_map& operator= (global_domain_map&& other)
                  {
                      localities_  = std::move(other.localities_);
                      index_set    = std::move(other.index_set);
                      localities_  = std::move(other.localities_);
                      return *this;
                  }
        // accessors
                  size_type size() const noexcept
                    {
                      return index_set.size();
                    }

                  bool empty() const noexcept
                    {
                      if(index_set.size() == 0)
                        return true;
                      else
                        return false;
                    }

                 void clear() noexcept
                    {
                      localities_.clear();
                      index_set.clear();
                      index_size = 0;
                    }

        //Distributive functions
            /*  do_cyclic() const
              {
                for(std::size_t i = 0; i < index_size; i++ )
                  {
                    for(auto const& id : localities_)
                      components_ids.emplace_back(hpx::new_<detail::local_domain_descriptor<T> >(id,index_set[i]));
                  }
              }
            */

            /*
              It needs working
            */
            void  do_cyclic() const
              {
                boost::atomic<std::size_t> inc(0);
                boost::atomic<std:;size_t> count(0);
                if(index_size >= localities_.size())
                {
                  for(auto const& id : localities_)
                    {
                     components_ids.emplace_back(hpx::new_<detail::local_domain_descriptor<T> >(id,index_set[inc]));
                      inc++; count++;
                    }
                }

                else
                {
                  for(auto const& index : index_set)
                    {
                      components_ids.emplace_back(hpx::new_<detail::local_domain_descriptor<T> >(localities_[inc],index));
                      count++;
                    }
                }

              }


              /*do_cyclic_block(std::size_t block_size) const
              {
                HPX_ASSERT(block_size < index_size);
                for(std::size_t j = 0; j < index_size; j++)
                  {
                    for(auto const& id : localities_ )
                      {

                      }
                  }
              }
                */
              /*
              do_block(hpx::id_type loc_ = localities_[0]) const
              {
                  = hpx::new_<detail::local_domain_descriptor<T> >(loc_,index_set);
              }
              */

            //this mapping function will work only for cyclic -Distribution policy
            // This fuction will be removed
            hpx::id_type mapping(T& index) const
            {
              using namespace hpx::parallel;
              auto it = hpx::parallel::find(par(task),index_set.begin(),index_set.end(),index);
              auto position = it.then([](auto i)
                              {
                                return std::distance(index_set.begin() - i.get());
                              });
              if(position <= localities_.size())
                return components_ids[position];
              else
                {
                  std::size_t pos = position % localities_.size();
                  return components_ids[pos];
                }
            }

            bool operator-= (const T& value)
            {
              using namespace hpx::parallel;
              auto fut_remove_index = hpx::parallel::find(par(task),
              index_set.begin(),index_set.end(),value);
              auto remove_index = fut_remove_index.get();
              if( remove_index != index_set.end())
                {
                  hpx::id_type gid_ = mapping(value);
                  auto end = hpx::async<remove_index_action>(gid,value);
                  index_set.erase(remove_index);
                  index_size --;
                  if(end.get() == false)
                    return false;
                  else
                  return true;
                }
                return false;
            }

            bool operator-= (T&& value)
            {
              using namespace hpx::parallel;
              auto fut_remove_index = hpx::parallel::find(par(task),
              index_set.begin(),index_set.end(),std::move(value));
              auto remove_index = fut_remove_index.get();
              if( remove_index != index_set.end())
                {
                  hpx::id_type gid_ = mapping(value);
                  auto end = hpx::async<remove_index_action>(gid,std::forward<T>(value));
                  index_set.erase(remove_index);
                  index_size --;
                  if(end.get() == false)
                    return false;
                  else
                  return true;
                }
                return false;
            }
            void operator+= (const T& value)
           {
              using namespace hpx::parallel;
              auto  available = hpx::parallel::find(par(task),
              index_set.begin(),index_set.end(),value);
              HPX_ASSERT(available.get() == index_set.end(),
                          "Duplicates not Allowed in Associative domains");
              auto end = hpx::async<add_index_action>(components_ids.back(),value);
              index_set.emplace_back(value);
              end.get();
           }

         void operator+= (T&& value)
          {
             using namespace hpx::parallel;
             auto  available = hpx::parallel::find(par(task),
             index_set.begin(),index_set.end(),std::move(value));
             HPX_ASSERT(available.get() == index_set.end(),
                         "Duplicates not Allowed in Associative domains");
             auto end = hpx::async<add_index_action>(components_ids.back(),std::forward<T>(value));
             index_set.emplace_back(std::move(value));
             end.get();
          }

            //Set-Associative Operations only support on non-const domains
            domain | operator(domain& other)
            {
              using namespace hpx::parallel;
              hpx::future<bool> res = hpx::parallel::is_sorted(par(task),
              other.index_set.begin(), other.index_set.end());

              auto te = res.then([other](hpx::future<bool> f1)
                                              {
                                               if(f1.get() != true){
                                                  auto f = hpx::parallel::sort(par(task),
                                                   other.index_set.begin(),other.index_set.end()).get();
                                                   return hpx::make_ready_future(1);
                                                  }
                                               else
                                                 return hpx::make_ready_future(1);
                                               }
                                              );
              hpx::future<bool> res_t  = hpx::parallel::is_sorted(par(task),
                  index_set.begin(),index_set.end());
              auto te_t  = res_t.then([this](hpx::future<bool> f2)
                                      {
                                        if(f2.get() != true){
                                           auto f = hpx::parallel::sort(par(task),
                                            index_set.begin(),index_set.end()).get();
                                            return hpx::make_ready_future(1);
                                           }
                                        else
                                          return hpx::make_ready_future(1);
                                      }
                                     );
                                     return hpx::dataflow(hpx::launch::sync,
                                      [this,other]  (auto &&value_1, auto &&value_2) -> domain
                                       {
                                          using namespace hpx::parallel;
                                          std::vector<T> dest;
                                          auto temp = hpx::parallel::set_union(par(task),
                                          this->index_set.begin(),this->index_set.end(),
                                          other.index_set.begin(),other.index_set.end(),
                                          dest.begin()); // check correctness
                                          return temp.then([&dest](auto && tem)
                                           {
                                             return global_domain_map(dest);
                                           }
                                           );
                                       },std::move(te), std::move(te_t)
                                     );}

        private:
                std::vector<hpx::id_type> localities_;
                std::vector<T>            index_set;
                std::size_t               index_size;
                std::vector<hpx::future<hpx::id_type> > components_ids;
                global_domain_map(const std::vector<T>& to_construct)
                  :
                    index_set(to_construct),
                    index_size(to_construct.size()),
                    localities_(hpx::find_here()) {}


      };

    template<>
    struct global_domain_map<arithmetic> : simple_component_base<global_domain_map<arithmetic> >
    {
      //  specialization for arithmetic domains

    };
