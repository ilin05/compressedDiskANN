#pragma once

#include <cstdint>
#include <cstring>

namespace utils {
    class BaseBlockStreamReader {
    public:
        virtual ~BaseBlockStreamReader() = default;
        virtual void cacheData(int beginBit, int endBit) = 0;
        virtual long long readLong(int size) = 0;
        virtual int readInt(int size) = 0;
        virtual void init() = 0;
        
        virtual bool readBoolean() {
            return readLong(1) > 0;
        }

        virtual float readFloat(int size) {
            uint32_t bits = static_cast<uint32_t>(readLong(size));
            float value;
            std::memcpy(&value, &bits, sizeof(float));
            return value;
        }

        virtual double readDouble(int size) {
            uint64_t bits = static_cast<uint64_t>(readLong(size));
            double value;
            std::memcpy(&value, &bits, sizeof(double));
            return value;
        }
    };
}
