#pragma once

#include "core/dtype.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

namespace ninfer {

struct DeviceSpan {
    void* data        = nullptr;
    std::size_t bytes = 0;
};

// Owning device allocation for long-lived buffers. DeviceArena remains the
// suballocation primitive for workspaces; this type owns exactly one cudaMalloc.
class DeviceBuffer {
public:
    DeviceBuffer() noexcept = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void fill(int byte_value = 0);
    void copy_from_host(const void* source, std::size_t count, std::size_t byte_offset = 0);
    void copy_to_host(void* destination, std::size_t count, std::size_t byte_offset = 0) const;

    // Raw access is intentional: Tensor and Weight are non-owning views.
    void* p           = nullptr;
    std::size_t bytes = 0;

private:
    void require_range(std::size_t byte_offset, std::size_t count, const char* operation) const;
};

class DeviceArena {
public:
    class Scope {
    public:
        ~Scope() noexcept;

        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept;
        Scope& operator=(Scope&&) = delete;

    private:
        friend class DeviceArena;

        explicit Scope(DeviceArena& arena) noexcept;

        DeviceArena* arena_       = nullptr;
        std::size_t saved_offset_ = 0;
    };

    explicit DeviceArena(std::size_t capacity_bytes);
    // Non-owning arena over an already allocated device region.
    explicit DeviceArena(DeviceSpan storage);
    // Protected default ctor for managed subclasses that perform their own allocation.
    DeviceArena() noexcept = default;
    virtual ~DeviceArena();

    DeviceArena(const DeviceArena&)            = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    DeviceArena(DeviceArena&& other) noexcept;
    DeviceArena& operator=(DeviceArena&& other) noexcept;

    virtual DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256);
    virtual Tensor alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align = 256);
    [[nodiscard]] virtual Scope scope() noexcept;
    virtual void reset() noexcept;

    virtual void* base() const noexcept;
    virtual std::size_t used() const noexcept;
    virtual std::size_t capacity() const noexcept;
    virtual std::size_t peak_used() const noexcept;
    virtual void reset_peak() noexcept;

protected:
    void* base_       = nullptr;
    std::size_t cap_  = 0;
    std::size_t off_  = 0;
    std::size_t peak_ = 0;
    bool owns_        = true;
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t size_bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() const noexcept;
    std::size_t size() const noexcept;

private:
    void* data_       = nullptr;
    std::size_t size_ = 0;
};

// Unified-memory arena for the context-scaling KV cache and (under --kv-managed) the model
// device state. Backed by cudaMallocManaged with a GPU-preferred location advise, so allocations
// may oversubscribe device VRAM and transparently page to host RAM (mirrors the Windows/WDDM
// residency model where the large context-scaling buffers are not pinned to physical VRAM).
// Inherits DeviceArena so existing DeviceArena& consumers dispatch to the managed implementation
// via virtual methods. NOT used for weights or hot decode workspaces unless --kv-managed is set
// (at which point the whole materialized arena oversubscribes, trading residency for capacity).
class ManagedDeviceArena : public DeviceArena {
public:
    explicit ManagedDeviceArena(std::size_t capacity_bytes);
    ~ManagedDeviceArena() override;

    ManagedDeviceArena(const ManagedDeviceArena&)            = delete;
    ManagedDeviceArena& operator=(const ManagedDeviceArena&) = delete;
    ManagedDeviceArena(ManagedDeviceArena&& other) noexcept;
    ManagedDeviceArena& operator=(ManagedDeviceArena&& other) noexcept;

    DeviceSpan alloc_bytes(std::size_t bytes, std::size_t align = 256) override;
    void* base() const noexcept override;
    std::size_t used() const noexcept override;
    std::size_t capacity() const noexcept override;
    std::size_t peak_used() const noexcept override;
    void reset() noexcept override;
    void reset_peak() noexcept override;
};

using WorkspaceArena = DeviceArena;

namespace core {
#if defined(_WIN32)
void set_wddm_residency_lock_enabled(bool enabled) noexcept;
[[nodiscard]] bool wddm_residency_lock_enabled() noexcept;
#endif
} // namespace core

} // namespace ninfer
