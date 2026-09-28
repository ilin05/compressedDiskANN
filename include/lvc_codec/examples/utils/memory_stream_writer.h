#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "base_stream_writer.h"

namespace utils {
    class MemoryStreamWriter : public BaseStreamWriter {
    private:
        std::vector<char>* vectorBuffer = nullptr;
        char* rawBuffer = nullptr;
        size_t rawCapacity = 0;
        size_t rawPos = 0;
        bool useVector = false;

        int currentByte = 0;
        int leftBits = 8;
        long long delta_bits = 0;

        void saveByte() {
            if (leftBits == 8) return;
            currentByte <<= leftBits; // padding with 0
            char byte = static_cast<char>(currentByte & 0xFF);
            
            if (useVector) {
                if (vectorBuffer) {
                    vectorBuffer->push_back(byte);
                }
            } else {
                if (rawBuffer && rawPos < rawCapacity) {
                    rawBuffer[rawPos++] = byte;
                } else {
                    throw std::runtime_error("MemoryStreamWriter: Buffer overflow");
                }
            }

            leftBits = 8;
            currentByte = 0;
        }

    public:
        explicit MemoryStreamWriter(std::vector<char>* buf) : vectorBuffer(buf), useVector(true) {
            init();
        }

        explicit MemoryStreamWriter(char* buf, size_t size) : rawBuffer(buf), rawCapacity(size), useVector(false) {
            init();
        }

        void init() {
            leftBits = 8;
            currentByte = 0;
            delta_bits = 0;
        }

        void setBuffer(char* buf, size_t size) {
            rawBuffer = buf;
            rawCapacity = size;
            rawPos = 0;
            useVector = false;
            init();
        }

        void setBuffer(std::vector<char>* buf) {
            vectorBuffer = buf;
            useVector = true;
            init();
        }

        long long track_bits() override {
            long long b = delta_bits;
            this->delta_bits = 0;
            return b;
        }

        void clear() override {
            if (currentByte == 0 && leftBits == 8) return;
            saveByte();
            init();
        }

        void write(bool b) override {
            delta_bits += 1;
            leftBits--;
            currentByte = (currentByte << 1) | (b ? 1 : 0);
            if (leftBits == 0) {
                saveByte();
            }
        }

        void write(long long value, int size) override {
            delta_bits += std::max(0, size);
            while (size > 0) {
                int len = std::min(size, leftBits);
                int mask = (1 << len) - 1;
                currentByte <<= len;
                currentByte |= (int)((value >> (size - len)) & mask);
                leftBits -= len;
                if (leftBits == 0) {
                    saveByte();
                }
                size -= len;
            }
        }

        void write(double value, int size) {
            union {
                double d;
                uint64_t l;
            } u;
            u.d = value;
            write(static_cast<long long>(u.l), size);
        }

        void write(int value, int size) override {
            write(static_cast<long long>(value), size);
        }

        int align() override {
            if(leftBits == 8) return 0;
            int padding = leftBits;
            saveByte();
            return padding;
        }

        int getLeftBits() const {
            return leftBits;
        }
    };
}
