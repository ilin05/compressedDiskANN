#pragma once

#include <cstdint>
#include <cstring>

namespace utils {
    class BaseStreamWriter {
    public:
        virtual ~BaseStreamWriter() = default;
        virtual void write(bool b) = 0;
        virtual void write(long long value, int size) = 0;
        virtual void write(int value, int size) = 0;

        virtual void write(float value, int size) {
            uint32_t i;
            std::memcpy(&i, &value, sizeof(float));
            write(static_cast<long long>(i), size);
        }

        virtual void write(double value, int size) {
            uint64_t l;
            std::memcpy(&l, &value, sizeof(double));
            write(static_cast<long long>(l), size);
        }

        virtual long long track_bits() = 0;
        virtual void clear() = 0;
        virtual int align() = 0;
    };
}
