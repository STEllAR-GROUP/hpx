if "%1%" == "" (
    echo Usage c:\path\to\boost\root
) else (
    cl -I %1% -I . /Zi /EHsc  libs\backtrace\src\backtrace.cpp libs\backtrace\test\test_backtrace.cpp /Fetest_backtrace.exe dbghelp.lib
)

