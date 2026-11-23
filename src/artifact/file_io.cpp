// Linux/POSIX implementation of the generic artifact InputFile. The upstream
// file_io.cpp on this branch is Windows-only and is excluded on Linux in
// src/CMakeLists.txt.
#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <cerrno>
#include <filesystem>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <span>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace ninfer::artifact {
namespace {

[[noreturn]] void fail_io(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

int fd_of(void* handle) { return static_cast<int>(reinterpret_cast<std::intptr_t>(handle)); }

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    const int fd = ::open(path_.c_str(), O_RDONLY);
    if (fd < 0) { fail_io(path_, "open"); }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        fail_io(path_, "fstat");
    }
    if (!S_ISREG(st.st_mode)) {
        ::close(fd);
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    handle_ = reinterpret_cast<void*>(static_cast<std::intptr_t>(fd));
    bytes_  = static_cast<std::uint64_t>(st.st_size);
}

InputFile::~InputFile() {
    if (direct_handle_) {
        ::close(fd_of(direct_handle_));
        direct_handle_ = nullptr;
    }
    if (handle_) {
        ::close(fd_of(handle_));
        handle_ = nullptr;
    }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    const int fd = fd_of(handle_);
    while (!destination.empty()) {
        const std::size_t count =
            std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const ssize_t got = ::pread(fd, destination.data(), count,
                                    static_cast<off_t>(offset));
        if (got < 0) {
            if (errno == EINTR) { continue; }
            fail_io(path_, "pread");
        }
        if (got == 0) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(got);
        destination = destination.subspan(static_cast<std::size_t>(got));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned direct read");
    }
    if (destination.empty()) { return 0; }
    const int fd = fd_of(handle_);
    const ssize_t got = ::pread(fd, destination.data(), destination.size(),
                                static_cast<off_t>(offset));
    if (got < 0) {
        if (errno == EINTR) { return 0; }
        fail_io(path_, "pread direct");
    }
    return static_cast<std::size_t>(got);
}

} // namespace ninfer::artifact
