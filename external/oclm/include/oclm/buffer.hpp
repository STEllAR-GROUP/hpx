
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_BUFFER_HPP
#define OCLM_BUFFER_HPP

#include <oclm/config.hpp>

namespace oclm {
    struct event
    {
        event() {}
        event(cl_event e)
            : e_(e)
        {}

        operator cl_event const &() const
        {
            return e_;
        }

        private:
            cl_event e_;
    };
}
#include <oclm/command_queue.hpp>

namespace oclm
{
    template <typename Src, typename Enable = void>
    struct make_buffer_impl
    {
        typedef void type;
        static type call(Src);
    };

    namespace result_of
    {
        template <typename Src>
        struct make_buffer
        {
            typedef typename make_buffer_impl<Src>::type type;
        };
    }

    template <typename Src>
    typename result_of::make_buffer<Src>::type
    make_buffer(Src & src)
    {
        return make_buffer_impl<Src>::call(src);
    }

    template <typename T, typename Policy>
    struct buffer
    {
        typedef T value_type;
        typedef Policy policy_type;

        buffer()
            : data_start(0)
            , data_size(0)
        {}

        buffer(buffer const & o)
            : data_start(o.data_start)
            , data_size(o.data_size)
            , mem(o.mem)
        {}

        template <typename U>
        explicit buffer(U & u)
        {
            typedef 
                typename result_of::make_buffer<U>::type
                src_buffer_type;

            // static_assert(is_convertible<typename src_buffer_type::value_type, value_type>::value, "")

            src_buffer_type src(make_buffer(u));
            (*this) = src;
        }
        
        template <typename U>
        buffer& operator=(U & u)
        {
            typedef 
                typename result_of::make_buffer<U>::type
                src_buffer_type;

            // static_assert(is_convertible<typename src_buffer_type::value_type, value_type>::value, "")

            src_buffer_type src(make_buffer(u));
            (*this) = src;
        }

        buffer & operator=(buffer const & o)
        {
            data_start = o.data_start;
            data_size = o.data_size;
            cl_mem mem;
        }

        void create(command_queue const & queue)
        {
            Policy::create(queue, *this);
        }

        void set_kernel_arg(cl_kernel k, std::size_t idx)
        {
            Policy::set_kernel_arg(k, idx, *this);
        }

        void write(command_queue const & queue)
        {
            std::vector<cl_event> tmp1;
            std::vector<cl_event> tmp2;
            write(queue, tmp1, tmp2);
        }

        void write(command_queue const & queue, cl_event const & e, std::vector<cl_event> & events)
        {
            std::vector<cl_event> tmp(1, e);
            write(queue, tmp, events);
        }

        void write(command_queue const & queue, std::vector<cl_event> & events)
        {
            std::vector<cl_event> tmp;
            write(queue, tmp, events);
        }
            
        void write(command_queue const & queue, std::vector<cl_event> & wait, std::vector<cl_event> &events)
        {
            Policy::write(queue, *this, wait, events);
        }

        void read(command_queue const & queue)
        {
            std::vector<cl_event> tmp;
            read(queue, tmp);
        }

        void read(command_queue const & queue, event const & e, std::vector<cl_event> & events)
        {
            std::vector<cl_event> tmp(1, e);
            read(queue, tmp, events);
        }

        void read(command_queue const & queue, std::vector<cl_event> & events)
        {
            std::vector<cl_event> tmp;
            read(queue, tmp, events);
        }
        
        void read(command_queue const & queue, std::vector<cl_event> & wait, std::vector<cl_event> &events)
        {
            Policy::read(queue, *this, wait, events);
        }

        T * data_start;
        std::size_t data_size;
        cl_mem mem;
    };

    namespace policy
    {
        struct noop
        {
            template <typename Buffer>
            static void create(command_queue const & queue, Buffer & b)
            {}

            template <typename Buffer>
            static void set_kernel_arg(cl_kernel k, std::size_t idx, Buffer & b)
            {
                cl_int err = CL_SUCCESS;
                err = clSetKernelArg(k, idx, sizeof(cl_mem), &b.mem);
                OCLM_THROW_IF_EXCEPTION(err, "clSetKernelArg");
            }

            template <typename Buffer>
            static void write(command_queue const &, Buffer &, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
            
            template <typename Buffer>
            static void read(command_queue const &, Buffer &, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
            
            template <typename Buffer1, typename Buffer2>
            static void copy(command_queue const &, Buffer1 &, Buffer2 &, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
        };
        
        struct io
        {
            template <typename Buffer>
            static void create(command_queue const & queue, Buffer & b)
            {
                cl_int err = CL_SUCCESS;
                b.mem = clCreateBuffer(queue.ctx_, CL_MEM_READ_WRITE, b.data_size, NULL, &err);
                OCLM_THROW_IF_EXCEPTION(err, "clCreateBuffer");
            }
            template <typename Buffer>
            static void set_kernel_arg(cl_kernel k, std::size_t idx, Buffer & b)
            {
                cl_int err = CL_SUCCESS;
                err = clSetKernelArg(k, idx, sizeof(cl_mem), &b.mem);
                OCLM_THROW_IF_EXCEPTION(err, "clSetKernelArg");
            }
            template <typename Buffer>
            static void write(command_queue const & queue, Buffer & b, std::vector<cl_event> & wait_list, std::vector<cl_event> &events)
            {
                cl_int err = CL_SUCCESS;
                cl_event event;
                err = clEnqueueWriteBuffer(queue, b.mem, false, 0, b.data_size , b.data_start, wait_list.size(), &wait_list[0], &event);
                OCLM_THROW_IF_EXCEPTION(err, "clEnqueueWriteBuffer");
                events.push_back(event);
            }
            
            template <typename Buffer>
            static void read(command_queue const & queue, Buffer & b, std::vector<cl_event> & wait_list, std::vector<cl_event> &events)
            {
                cl_int err = CL_SUCCESS;
                cl_event event;
                err = clEnqueueReadBuffer(queue, b.mem, false, 0, b.data_size, b.data_start, wait_list.size(), &wait_list[0], &event);
                OCLM_THROW_IF_EXCEPTION(err, "clEnqueueReadBuffer");
                events.push_back(event);
            }
            
            template <typename Buffer1, typename Buffer2>
            static void copy(command_queue const & q, Buffer1 & src, Buffer2 & dst, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
        };

        struct input
        {
            template <typename Buffer>
            static void create(command_queue const & queue, Buffer & b)
            {
                cl_int err = CL_SUCCESS;
                b.mem = clCreateBuffer(queue.ctx_, CL_MEM_READ_WRITE, b.data_size, NULL, &err);
                OCLM_THROW_IF_EXCEPTION(err, "clCreateBuffer");
            }
            template <typename Buffer>
            static void set_kernel_arg(cl_kernel k, std::size_t idx, Buffer & b)
            {
                cl_int err = CL_SUCCESS;
                err = clSetKernelArg(k, idx, sizeof(cl_mem), &b.mem);
                OCLM_THROW_IF_EXCEPTION(err, "clSetKernelArg");
            }

            template <typename Buffer>
            static void write(command_queue const & queue, Buffer & b, std::vector<cl_event> & wait_list, std::vector<cl_event> &events)
            {
                cl_int err = CL_SUCCESS;
                cl_event event;
                err = clEnqueueWriteBuffer(queue.ctx_, b.mem, false, 0, b.data_size , b.data_start, wait_list.size(), &wait_list[0], &event);
                OCLM_THROW_IF_EXCEPTION(err, "clEnqueueWriteBuffer");
                events.push_back(event);
            }
            
            
            template <typename Buffer>
            static void read(command_queue const &, Buffer &, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
            
            template <typename Buffer1, typename Buffer2>
            static void copy(command_queue const &, Buffer1 &, Buffer2 &, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
        };

        struct output
        {
            template <typename Buffer>
            static void create(command_queue const & queue, Buffer & b)
            {
                cl_int err = CL_SUCCESS;
                b.mem = clCreateBuffer(queue.ctx_, CL_MEM_READ_WRITE, b.data_size, NULL, &err);
                OCLM_THROW_IF_EXCEPTION(err, "clCreateBuffer");
            }
            template <typename Buffer>
            static void set_kernel_arg(cl_kernel k, std::size_t idx, Buffer & b)
            {
                cl_int err = CL_SUCCESS;
                err = clSetKernelArg(k, idx, sizeof(cl_mem), &b.mem);
                OCLM_THROW_IF_EXCEPTION(err, "clSetKernelArg");
            }

            template <typename Buffer>
            static void write(command_queue const &, Buffer &, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
            
            template <typename Buffer>
            static void read(command_queue const & queue, Buffer & b, std::vector<cl_event> & wait_list, std::vector<cl_event> &events)
            {
                cl_int err = CL_SUCCESS;
                cl_event event;
                err = clEnqueueReadBuffer(queue, b.mem, false, 0, b.data_size, b.data_start, wait_list.size(), &wait_list[0], &event);
                OCLM_THROW_IF_EXCEPTION(err, "clEnqueueReadBuffer");
                events.push_back(event);
            }
            
            template <typename Buffer1, typename Buffer2>
            static void copy(command_queue const &, Buffer1 &, Buffer2 &, std::vector<cl_event> &, std::vector<cl_event> &)
            {}
        };
    }

    template <typename T>
    struct make_buffer_impl<std::vector<T> >
    {
        typedef buffer<T, policy::io> type;

        static type call(std::vector<T> &t)
        {
            type res;
            res.data_start = &t[0];
            res.data_size = t.size() * sizeof(T);
            return res;
        }
    };
}

#endif
