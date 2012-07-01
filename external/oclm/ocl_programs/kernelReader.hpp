#include <fstream>
#include <string>

bool readKernel(const char* fileName, size_t& source_size, char *source_str, const int MAX_SOURCE_SIZE)
{
    FILE *fp;
    std::string path = __FILE__;
    std::string cl_path;
    std::string remove = "kernelReader.hpp";
    for (int i = 0; i < path.length() - remove.length(); i++)
    {
        cl_path.push_back(path[i]);
        if (path[i] == '\\')
        {
            cl_path.push_back(path[i]);
        }
    }
    cl_path += fileName;
    fp = fopen(cl_path.c_str(), "r");

    if (!fp) {
        fprintf(stderr, cl_path.c_str());
        fprintf(stderr, "\n");
        return false;
    }
    char* source_local = new char[MAX_SOURCE_SIZE];
    source_size = fread( source_local, 1, MAX_SOURCE_SIZE, fp );
    strcpy(source_str, source_local);
    fclose( fp );
    return true;
}