// Compatibility definitions for graph construction and entry generation.
#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

template <typename T, std::size_t Alignment = 32>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        if (n > std::size_t(-1) / sizeof(T)) {
            throw std::bad_alloc();
        }

        void* ptr = nullptr;
        size_type bytes = n * sizeof(T);

        int result = posix_memalign(&ptr, Alignment, bytes);

        if (result != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer ptr, size_type n) noexcept {
        (void)n;
        std::free(ptr);
    }

    template <typename U, std::size_t A>
    bool operator==(const AlignedAllocator<U, A>&) const noexcept {
        return Alignment == A;
    }

    template <typename U, std::size_t A>
    bool operator!=(const AlignedAllocator<U, A>&) const noexcept {
        return Alignment != A;
    }
};

template <typename T>
using AlignedVector32 = std::vector<T, AlignedAllocator<T, 32>>;

template <typename T>
using AlignedVector64 = std::vector<T, AlignedAllocator<T, 64>>;

#endif
