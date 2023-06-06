#include <hpx/type_support/unused.hpp>

#include <cstddef>
#include <string>

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__)) &&               \
    !defined(HPX_MINGW)

#include <windows.h>

void detail_set_thread_name_helper(char const* thread_name) noexcept
{
    // set thread name the 'old' way
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = thread_name;
    info.dwThreadID = -1;
    info.dwFlags = 0;

    __try
    {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR),
            reinterpret_cast<ULONG_PTR*>(&info));
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
    }
}

std::wstring to_wide_string(char const* name) noexcept
{
    try
    {
        std::string const str(name);
        if (str.empty())
            return {};

        // determine required length of new string
        std::size_t const req_length = MultiByteToWideChar(CP_UTF8, 0,
            str.c_str(), static_cast<int>(str.length()), nullptr, 0);

        // construct new string of required length
        std::wstring ret(req_length, L'\0');

        // convert old string to new string
        MultiByteToWideChar(CP_UTF8, 0, str.c_str(),
            static_cast<int>(str.length()), ret.data(),
            static_cast<int>(ret.length()));

        return ret;
    }
    catch (...)
    {
        return {};
    }
}

detail_set_thread_name(const char* thread_name)
{
    detail_set_thread_name_helper(thread_name);
    SetThreadDescription(
        GetCurrentThread(), to_wide_string(thread_name).c_str());
}
#elif (defined(__linux__))

#include <pthread.h>

void detail_set_thread_name(char const* thread_name)
{
    pthread_setname_np(pthread_self(), thread_name);
}

int detail_get_thread_name(char* buf, size_t len)
{
    return pthread_getname_np(pthread_self(), buf, len);
}

#endif

int main()
{
    detail_set_thread_name("hpx_thread_1");
}
