#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include "base_block_stream_reader.h"

namespace utils {

    class MemoryBlockStreamReader : public BaseBlockStreamReader {
    private:
        const unsigned char* data_source;
        std::vector<unsigned char> buffer;
        size_t cachedBytes = 0;
        size_t pointer = 0;
        int leftBits = 8;

    public:
        MemoryBlockStreamReader(const unsigned char* data)
            : data_source(data), buffer(1024, 0) {}

        void init() override {
            pointer = 0;
            leftBits = 8;
        }

        void cacheData(int beginBit, int endBit) override {
            if (endBit <= beginBit) {
                cachedBytes = 0;
                init();
                return;
            }

            size_t byteBegin = static_cast<size_t>(beginBit) / 8;
            size_t byteEnd = (static_cast<size_t>(endBit) + 7) / 8; // endBit exclusive
            size_t bytesToRead = (byteEnd > byteBegin) ? (byteEnd - byteBegin) : 0;

            if (bytesToRead == 0) {
                cachedBytes = 0;
                init();
                return;
            }

            if (buffer.size() < bytesToRead) {
                buffer.resize(bytesToRead);
            }

            std::memcpy(buffer.data(), data_source + byteBegin, bytesToRead);
            cachedBytes = bytesToRead;

            int offset = beginBit % 8;
            if (offset != 0 && cachedBytes > 0) {
                for (size_t i = 0; i < cachedBytes; ++i) {
                    unsigned char curr = buffer[i];
                    unsigned char next = (i + 1 < cachedBytes) ? buffer[i + 1] : 0;
                    buffer[i] = static_cast<unsigned char>(((curr << offset) & 0xFFu) | (next >> (8 - offset)));
                }
            }

            init();
        }

        void resetBuffer(const unsigned char* ptr, size_t size) {
            if (buffer.size() < size) {
                buffer.resize(size);
            }
            std::memcpy(buffer.data(), ptr, size);
            cachedBytes = size;
            init();
        }

        long long readLong(int size) override {
            long long result = 0;
            while (size > 0 && pointer < cachedBytes) {
                int bitsToRead = std::min(leftBits, size);
                leftBits -= bitsToRead;
                unsigned int mask = (1u << bitsToRead) - 1u;
                unsigned int chunk = (buffer[pointer] >> leftBits) & mask;
                result = (result << bitsToRead) | static_cast<long long>(chunk);
                size -= bitsToRead;
                if (leftBits == 0) {
                    pointer++;
                    leftBits = 8;
                }
            }

            if (size > 0) {
                result <<= size; // pad remaining bits with zeros when data exhausted
            }
            return result;
        }

        int readInt(int size) override {
            return static_cast<int>(readLong(size));
        }

        bool readBoolean() override {
            return readLong(1) > 0;
        }

        double readDouble(int size) override {
            std::uint64_t bits = static_cast<std::uint64_t>(readLong(size));
            double value;
            std::memcpy(&value, &bits, sizeof(double));
            return value;
        }
    };
}
