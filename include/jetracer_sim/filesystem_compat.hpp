#pragma once

#include <system_error>

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ < 8
#include <experimental/filesystem>
namespace jetracer_filesystem = std::experimental::filesystem;
#define JETRACER_EXPERIMENTAL_FILESYSTEM 1
#else
#include <filesystem>
namespace jetracer_filesystem = std::filesystem;
#endif

namespace jetracer_filesystem_compat {

inline jetracer_filesystem::path normalized(
    const jetracer_filesystem::path& path) {
#ifdef JETRACER_EXPERIMENTAL_FILESYSTEM
  return path;
#else
  return path.lexically_normal();
#endif
}

inline jetracer_filesystem::path relative(
    const jetracer_filesystem::path& path,
    const jetracer_filesystem::path& base, std::error_code& error) {
#ifdef JETRACER_EXPERIMENTAL_FILESYSTEM
  (void)base;
  error.clear();
  return path;
#else
  return jetracer_filesystem::relative(path, base, error);
#endif
}

}  // namespace jetracer_filesystem_compat
