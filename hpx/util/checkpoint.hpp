// Copyright (c) 2017 Adrian Serio
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
/// This header defines the save_checkpoint and restore_checkpoint functions. 
/// These functions are designed to help HPX application developers checkpoint 
/// thier applications. Save_checkpoint serializes one or more objects and saves 
/// them as a byte stream. Restore_checkpoint converts the byte stream back into 
/// instances of the objects.
//

/// \file hpx/util/checkpoint.hpp

#if !defined(CHECKPOINT_HPP_07262017)
#define CHECKPOINT_HPP_07262017

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/lcos/future.hpp>

#include <cstddef>
#include <fstream>
#include <iosfwd>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx
{
namespace util
{

// Forward declarations
class checkpoint;
std::ostream& operator<<(std::ostream& ost, checkpoint const& ckp);
std::istream& operator>>(std::istream& ist, checkpoint& ckp);
namespace detail
{
struct save_funct_obj;
}

    ///////////////////////////////////
    /// Checkpoint Object
    ///
    /// Checkpoint is the container object which is produced by save_checkpoint 
    /// and is consumed by a restore_checkpoint. A checkpoint may be moved into
    /// the save_checkpoint object to write the byte stream to the pre-created 
    /// checkpoint object.
    class checkpoint
    {
        std::vector<char> data;

        friend std::ostream& operator<<(std::ostream& ost, checkpoint const& ckp);
        friend std::istream& operator>>(std::istream& ist, checkpoint& ckp);
        //Serialization Definition
        friend class hpx::serialization::access;
        template <typename Archive>
        void serialize(Archive& arch, const unsigned int version)
        {
            arch& data;
        };
        friend struct detail::save_funct_obj;
        template <typename T, typename... Ts>
        friend void restore_checkpoint(checkpoint const& c, T& t, Ts& ... ts);
        
    public:
        checkpoint() = default;
        checkpoint(checkpoint const& c)
          : data(c.data)
        {
        }
        checkpoint(checkpoint&& c)
         : data(std::move(c.data))
        {
        }
        ~checkpoint() = default;

        //Other Constructors
        checkpoint(char* stream, std::size_t count)
        {
            for (std::size_t i=0;i<count;i++)
            {
                data.push_back(*stream);
                stream++;
            }
        }

        checkpoint& operator=(checkpoint const& c)
        {
            if (&c!=this)
            {
                data = c.data;
            }
            return *this;
        }
        checkpoint& operator=(checkpoint&& c)
        {
            if (&c!=this)
            {
                data = std::move(c.data);
            }
            return *this;
        }

        bool operator==(checkpoint const& c) const
        {
            return data == c.data;
        }
        bool operator!=(checkpoint const& c) const
        {
            return !(data == c.data);
        }
        
        // Expose iterators to access data held by checkpoint
        using const_iterator = std::vector<char>::const_iterator;
        const_iterator begin() const
        {
            return data.begin();
        }
        const_iterator end() const
        {
            return data.end();
        }
/*
        void load(std::string file_name)
        {
            std::ifstream ifs(file_name);
            if(ifs)                               //Check fstream is open
            {
                ifs.seekg(0, ifs.end);
                int length = ifs.tellg();         //Get length of file
                ifs.seekg(0, ifs.beg);
                data.resize(length);
                ifs.read(data.data(), length);
            }
        }
*/
        size_t size() const
        {
            return data.size();
        }

    };

    //Stream Overloads
    std::ostream& operator<<(std::ostream& ost, checkpoint const& ckp)
    {
        // Write the size of the checkpoint to the file
        int64_t size = ckp.size();
        ost.write(reinterpret_cast<char const *>(&size), sizeof(int64_t));
        // Write the file to the stream
        ost.write(ckp.data.data(), ckp.size());
        return ost;
    }
    std::istream& operator>>(std::istream& ist, checkpoint& ckp)
    {
        // Read in the size of the next checkpoint
        int64_t length;
        ist.read(reinterpret_cast<char *>(&length), sizeof(int64_t));
        ckp.data.resize(length);
        // Read in the next checkpoint
        ist.read(ckp.data.data(), length);
        return ist;
    }

    //Function object for save_checkpoint
    namespace detail
    {
        struct save_funct_obj
     {
         template <typename... Ts>
         checkpoint operator()(checkpoint&& c, Ts&&... ts) const
         {
             //Create serialization archive from checkpoint data member
             hpx::serialization::output_archive ar(c.data);
             //Serialize data
             int const sequencer[] = { //Trick to expand the variable pack
                 (ar << ts, 0)...};    //Takes advantage of the comma operator
             (void)sequencer;          // Suppress unused param. warnings
             return c;
         }
     };
    }

    ///////////////////////////////////
    /// Save_checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be 
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint 
    ///                      to be serialized and placed into a 
    ///                      checkpoint object.
    ///                      
    /// \tparam U            This parameter is used to make sure that T
    ///                      is not a launch policy or a checkpoint. This
    ///                      forces the compiler to choose the correct overload.
    /// 
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes a any number of objects which a user may wish 
    /// to store and returns a future to a checkpoint object. 
    /// Additionally the function can take a policy as a first object which 
    /// changes its behaviour depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a 
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one 
    ///          exeption: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpiont will simply return
    ///          a checkpoint.

    template <typename T
            , typename... Ts
            , typename U = typename std::enable_if<
                    !hpx::traits::is_launch_policy<T>::value && 
                    !std::is_same<typename std::decay<T>::type,checkpoint>::value
                >::type
             >
    hpx::future<checkpoint> save_checkpoint(T&& t, Ts&&... ts)
    {
        {
            return hpx::dataflow(
                detail::save_funct_obj()
              , std::move(checkpoint())
              , std::forward<T>(t)
              , std::forward<Ts>(ts)...);
        }
    }
    
    ///////////////////////////////////
    /// Save_checkpoint - Take a pre-initialized checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be 
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint 
    ///                      to be serialized and placed into a 
    ///                      checkpoint object.
    /// 
    /// \param c             Takes a pre-initialized checkpint to copy
    ///                      data into.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes a any number of objects which a user may wish 
    /// to store and returns a future to a checkpoint object. 
    /// Additionally the function can take a policy as a first object which 
    /// changes its behaviour depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a 
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one 
    ///          exeption: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpiont will simply return
    ///          a checkpoint.

    template <typename T, typename... Ts>
    hpx::future<checkpoint> save_checkpoint(
          checkpoint&& c
        , T&& t
        , Ts&&... ts)
    {
        {
            return hpx::dataflow(
                detail::save_funct_obj()
              , std::move(c)
              , std::forward<T>(t)
              , std::forward<Ts>(ts)...);
        }
    }
    
    ///////////////////////////////////
    /// Save_checkpoint - Policy overload
    ///
    /// \tparam T            Containers passed to save_checkpoint to be 
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint 
    ///                      to be serialized and placed into a 
    ///                      checkpoint object.
    /// 
    /// \param p             Takes an HPX launch policy. Allows the user
    ///                      to change the way the function is launched
    ///                      i.e. async, sync, etc.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes a any number of objects which a user may wish 
    /// to store and returns a future to a checkpoint object. 
    /// Additionally the function can take a policy as a first object which 
    /// changes its behaviour depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a 
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one 
    ///          exeption: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpiont will simply return
    ///          a checkpoint.

    template <typename T, typename... Ts>
    hpx::future<checkpoint> save_checkpoint(
          hpx::launch p
        , T&& t
        , Ts&&... ts)
    {
        {
            return hpx::dataflow(
                  p
                , detail::save_funct_obj()
                , std::move(checkpoint())
                , std::forward<T>(t)
                , std::forward<Ts>(ts)...);
        }
    }

    ///////////////////////////////////
    /// Save_checkpoint - Policy overload & pre-initialized checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be 
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint 
    ///                      to be serialized and placed into a 
    ///                      checkpoint object.
    /// 
    /// \param p             Takes an HPX launch policy. Allows the user
    ///                      to change the way the function is launched
    ///                      i.e. async, sync, etc.
    ///
    /// \param c             Takes a pre-initialized checkpint to copy
    ///                      data into.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes a any number of objects which a user may wish 
    /// to store and returns a future to a checkpoint object. 
    /// Additionally the function can take a policy as a first object which 
    /// changes its behaviour depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a 
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint returns a future to a checkpoint with one 
    ///          exeption: if you pass hpx::launch::sync as the first
    ///          argument. In this case save_checkpiont will simply return
    ///          a checkpoint.

    template <typename T, typename... Ts>
    hpx::future<checkpoint> save_checkpoint(
          hpx::launch p
        , checkpoint&& c
        , T&& t
        , Ts&&... ts)
    {
        {
            return hpx::dataflow(
                  p
                , detail::save_funct_obj()
                , std::move(c)
                , std::forward<T>(t)
                , std::forward<Ts>(ts)...);
        }
    }

    ///////////////////////////////////
    /// Save_checkpoint - Sync_policy overload
    ///
    /// \tparam T            Containers passed to save_checkpoint to be 
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint 
    ///                      to be serialized and placed into a 
    ///                      checkpoint object.
    /// 
    /// \tparam U            This parameter is used to make sure that T
    ///                      is not a checkpoint. This forces the compiler 
    ///                      to choose the correct overload.
    /// 
    /// \param sync_p        hpx::launch::sync_policy
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes a any number of objects which a user may wish 
    /// to store and returns a future to a checkpoint object. 
    /// Additionally the function can take a policy as a first object which 
    /// changes its behaviour depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a 
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint which is passed hpx::launch::sync_policy 
    ///          will return a checkpoint which contains the serialized
    ///          values checkpoint.

    template <typename T
            , typename... Ts
            , typename U = typename std::enable_if<
                    !std::is_same<typename std::decay<T>::type,checkpoint>::value
                >::type
             >
    checkpoint save_checkpoint(
          hpx::launch::sync_policy sync_p
        , T&& t
        , Ts&&... ts)
    {
        {
            hpx::future<checkpoint> f_chk = 
                hpx::dataflow(
                      sync_p
                    , detail::save_funct_obj()
                    , std::move(checkpoint())
                    , std::forward<T>(t)
                    , std::forward<Ts>(ts)...);
            return f_chk.get();
        }
    }

    ///////////////////////////////////
    /// Save_checkpoint - Sync_policy overload & pre-init. checkpoint
    ///
    /// \tparam T            Containers passed to save_checkpoint to be 
    ///                      serialized and placed into a checkpoint object.
    ///
    /// \tparam Ts           More containers passed to save_checkpoint 
    ///                      to be serialized and placed into a 
    ///                      checkpoint object.
    /// 
    /// \param sync_p        hpx::launch::sync_policy
    ///
    /// \param c             Takes a pre-initialized checkpint to copy
    ///                      data into.
    ///
    /// \param t             A container to restore.
    ///
    /// \param ts            Other containers to restore Containers
    ///                      must be in the same order that they were
    ///                      inserted into the checkpoint.
    ///
    /// Save_checkpoint takes a any number of objects which a user may wish 
    /// to store and returns a future to a checkpoint object. 
    /// Additionally the function can take a policy as a first object which 
    /// changes its behaviour depending on the policy passed to it. Most
    /// notably, if a sync policy is used save_checkpoint will simply return a 
    /// checkpoint object.
    ///
    /// \returns Save_checkpoint which is passed hpx::launch::sync_policy 
    ///          will return a checkpoint which contains the serialized
    ///          values checkpoint.
    template <typename T, typename... Ts>
    checkpoint save_checkpoint(
          hpx::launch::sync_policy sync_p
        , checkpoint&& c
        , T&& t
        , Ts&&... ts)
    {
        {
            hpx::future<checkpoint> f_chk = 
                hpx::dataflow(
                      sync_p
                    , detail::save_funct_obj()
                    , std::move(c)
                    , std::forward<T>(t)
                    , std::forward<Ts>(ts)...);
            return f_chk.get();
        }
    }

    ///////////////////////////////////
    /// Resurrect
    ///
    /// Restore_checkpoint takes a checkpoint object as a first argument and the 
    /// containers which will be filled from the byte stream (in the same order
    /// as they were placed in save_checkpoint).
    ///
    /// \tparam T           A container to restore.
    ///                     
    /// \tparam Ts          Other continters to restore. Containers 
    ///                     must be in the same order that they were
    ///                     inserted into the checkpiont.
    ///                     
    /// \param c            The checkpoint to restore.
    ///
    /// \param t            A container to restore.
    ///
    /// \param ts           Other containers to restore Containers
    ///                     must be in the same order that they were
    ///                     inserted into the checkpoint.
    /// 
    /// \returns Restore_checkpoint returns void.
    template <typename T, typename... Ts>
    void restore_checkpoint(checkpoint const& c, T& t, Ts& ... ts)
    {
        {
            //Create seriaalization archive
            hpx::serialization::input_archive ar(c.data, c.size());
    
            //De-serialize data
            ar >> t;
            int const sequencer[] = {//Trick to exand the variable pack
               (ar >> ts, 0)...};    //Takes advantage of the comma operator
            (void)sequencer;         //Suppress unused param. warnings
        }
    }

} //End Util Namespace
} //End HPX Namespace

#endif

