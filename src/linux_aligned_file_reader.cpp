// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <cstring>
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 1024

namespace
{
void execute_io(IOContext ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0)
{
#ifdef DEBUG
    for (auto &req : read_reqs)
    {
        assert(IS_ALIGNED(req.len, 512));
        assert(IS_ALIGNED(req.offset, 512));
        assert(IS_ALIGNED(req.buf, 512));
    }
#endif

    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++)
    {
        uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
        for (uint64_t j = 0; j < n_ops; j++)
        {
            struct io_uring_sqe *sqe = io_uring_get_sqe(ctx);
            if (!sqe) {
                io_uring_submit(ctx);
                sqe = io_uring_get_sqe(ctx);
            }
            io_uring_prep_read(sqe, fd, read_reqs[j + iter * MAX_EVENTS].buf, read_reqs[j + iter * MAX_EVENTS].len, read_reqs[j + iter * MAX_EVENTS].offset);
            io_uring_sqe_set_data(sqe, &read_reqs[j + iter * MAX_EVENTS]);
        }

        io_uring_submit(ctx);

        for (uint64_t j = 0; j < n_ops; j++)
        {
            struct io_uring_cqe *cqe;
            int ret = io_uring_wait_cqe(ctx, &cqe);
            if (ret < 0) {
                std::cerr << "io_uring_wait_cqe failed: " << strerror(-ret) << std::endl;
                exit(-1);
            }
            if (cqe->res < 0) {
                std::cerr << "io_uring read failed: " << strerror(-cqe->res) << std::endl;
                exit(-1);
            }
            io_uring_cqe_seen(ctx, cqe);
        }
    }
}
} // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader()
{
    this->file_desc = -1;
}

LinuxAlignedFileReader::~LinuxAlignedFileReader()
{
    int64_t ret;
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            std::cerr << "close() not called" << std::endl;
            ret = ::close(this->file_desc);
            if (ret == -1)
            {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno) << std::endl;
            }
        }
    }
}

IOContext &LinuxAlignedFileReader::get_ctx()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end())
    {
        std::cerr << "bad thread access; returning nullptr as IOContext" << std::endl;
        return this->bad_ctx;
    }
    else
    {
        return ctx_map[std::this_thread::get_id()];
    }
}

void LinuxAlignedFileReader::register_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(my_id) != ctx_map.end())
    {
        std::cerr << "multiple calls to register_thread from the same thread" << std::endl;
        return;
    }
    
    struct io_uring* ring = new struct io_uring;
    // Try to setup with SQPOLL for best performance
    struct io_uring_params params;
    memset(&params, 0, sizeof(params));
    params.flags |= IORING_SETUP_SQPOLL;
    params.sq_thread_idle = 2000;
    
    int ret = io_uring_queue_init_params(MAX_EVENTS, ring, &params);
    if (ret != 0)
    {
        // Fallback to normal io_uring if SQPOLL is not supported
        ret = io_uring_queue_init(MAX_EVENTS, ring, 0);
        if (ret != 0) {
            lk.unlock();
            std::cerr << "io_uring_queue_init() failed; returned " << ret << ": " << ::strerror(-ret) << std::endl;
            delete ring;
            return;
        }
    }

    diskann::cout << "allocating ctx: " << ring << " to thread-id:" << my_id << std::endl;
    ctx_map[my_id] = ring;
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    assert(ctx_map.find(my_id) != ctx_map.end());

    IOContext ctx = ctx_map[my_id];
    io_uring_queue_exit(ctx);
    delete ctx;
    ctx_map.erase(my_id);
    std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    for (auto x = ctx_map.begin(); x != ctx_map.end(); x++)
    {
        IOContext ctx = x.value();
        io_uring_queue_exit(ctx);
        delete ctx;
    }
    ctx_map.clear();
}

void LinuxAlignedFileReader::open(const std::string &fname)
{
    int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
    this->file_desc = ::open(fname.c_str(), flags);
    assert(this->file_desc != -1);
    std::cerr << "Opened file : " << fname << std::endl;
}

void LinuxAlignedFileReader::close()
{
    ::fcntl(this->file_desc, F_GETFD);
    ::close(this->file_desc);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async)
{
    if (async == true)
    {
        diskann::cout << "Async currently not supported in linux." << std::endl;
    }
    assert(this->file_desc != -1);
    execute_io(ctx, this->file_desc, read_reqs);
}

void LinuxAlignedFileReader::submit_req(IOContext &ctx, std::vector<AlignedRead*> &read_reqs)
{
    uint64_t n_ops = read_reqs.size();
    if (n_ops == 0) return;

    for (uint64_t j = 0; j < n_ops; j++)
    {
        struct io_uring_sqe *sqe = io_uring_get_sqe(ctx);
        if (!sqe) {
            io_uring_submit(ctx);
            sqe = io_uring_get_sqe(ctx);
        }
        io_uring_prep_read(sqe, this->file_desc, read_reqs[j]->buf, read_reqs[j]->len, read_reqs[j]->offset);
        io_uring_sqe_set_data(sqe, read_reqs[j]);
    }

    io_uring_submit(ctx);
}

int LinuxAlignedFileReader::get_events(IOContext &ctx, int min_nr, int max_nr, std::vector<AlignedRead*> &completed_reqs)
{
    struct io_uring_cqe *cqe;
    unsigned head;
    int count = 0;

    // We can do a peek first to quickly get completed events, or wait for min_nr
    if (min_nr > 0) {
        int ret = io_uring_wait_cqe_nr(ctx, &cqe, min_nr);
        if (ret < 0) {
            std::cerr << "io_uring_wait_cqe_nr() failed; returned " << ret << ", errno=" << ::strerror(-ret) << "\n";
            exit(-1);
        }
    }

    io_uring_for_each_cqe(ctx, head, cqe) {
        if (count >= max_nr) break;
        
        if (cqe->res < 0) {
            std::cerr << "io_uring read failed: " << strerror(-cqe->res) << std::endl;
            exit(-1);
        }
        
        completed_reqs.push_back(static_cast<AlignedRead*>(io_uring_cqe_get_data(cqe)));
        count++;
    }

    io_uring_cq_advance(ctx, count);
    return count;
}
