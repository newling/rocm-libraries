// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Multi-process coverage of the fused-A2A path, plus the launcher-environment
// check that decides a rank's device. Each rank is a separate process, so the
// peers reach each other through hipIpcOpenMemHandle, which the ranks of a
// single process never do.
//
// The suffix on each suite name is the ctest category token.

#include "a2a_rank_child.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

extern char** environ;

namespace
{
    constexpr uint32_t kLaunches       = 4;
    constexpr int      kRankTimeoutSec = 180;
    constexpr int      kReapTimeoutSec = 10;

    const char* const kRankVariables[] = {"RANK=",
                                          "LOCAL_RANK=",
                                          "WORLD_SIZE=",
                                          "MASTER_ADDR=",
                                          "MASTER_PORT=",
                                          "A2A_LAUNCHES=",
                                          "A2A_COLLECTIVE="};

    bool setsRankIdentity(const char* entry)
    {
        for(const char* prefix : kRankVariables)
            if(strncmp(entry, prefix, strlen(prefix)) == 0)
                return true;
        return false;
    }

    int freeLoopbackPort()
    {
        const int probe = socket(AF_INET, SOCK_STREAM, 0);
        if(probe < 0)
            return -1;

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;

        socklen_t length = sizeof(addr);
        int       port   = -1;
        if(bind(probe, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0
           && getsockname(probe, reinterpret_cast<sockaddr*>(&addr), &length) == 0)
            port = ntohs(addr.sin_port);

        close(probe);
        return port;
    }

    int spawnRank(uint32_t rank, uint32_t world, int port, const char* collective, pid_t& pid)
    {
        const std::vector<std::string> assignments = {
            "RANK=" + std::to_string(rank),
            "LOCAL_RANK=" + std::to_string(rank),
            "WORLD_SIZE=" + std::to_string(world),
            "MASTER_ADDR=127.0.0.1",
            "MASTER_PORT=" + std::to_string(port),
            "A2A_LAUNCHES=" + std::to_string(kLaunches),
            std::string("A2A_COLLECTIVE=") + collective,
        };

        std::vector<char*> envp;
        for(const std::string& assignment : assignments)
            envp.push_back(const_cast<char*>(assignment.c_str()));
        for(char** entry = environ; *entry != nullptr; ++entry)
            if(!setsRankIdentity(*entry))
                envp.push_back(*entry);
        envp.push_back(nullptr);

        char* const argv[] = {const_cast<char*>("hipblaslt-test"),
                              const_cast<char*>("--a2a-rank-child"),
                              nullptr};

        return posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, argv, envp.data());
    }

    // Gives up on a rank that SIGKILL cannot reach.
    void killAll(const std::vector<pid_t>& pids)
    {
        for(pid_t pid : pids)
            if(pid > 0)
                kill(pid, SIGKILL);

        const auto deadline
            = std::chrono::steady_clock::now() + std::chrono::seconds(kReapTimeoutSec);
        for(pid_t pid : pids)
        {
            if(pid <= 0)
                continue;

            int discarded = 0;
            while(waitpid(pid, &discarded, WNOHANG) == 0
                  && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    // False when the deadline passes with a rank still running.
    bool waitForRanks(std::vector<pid_t>& pids, std::vector<int>& codes, int timeoutSec)
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeoutSec);
        codes.assign(pids.size(), -1);

        size_t exited = 0;
        for(size_t i = 0; i < pids.size(); ++i)
            if(pids[i] <= 0)
            {
                codes[i] = hipblaslt_bench::kRankChildFailed;
                ++exited;
            }

        while(exited < pids.size())
        {
            for(size_t i = 0; i < pids.size(); ++i)
            {
                if(pids[i] <= 0)
                    continue;

                int         status = 0;
                const pid_t seen   = waitpid(pids[i], &status, WNOHANG);
                if(seen != pids[i])
                    continue;

                codes[i] = WIFEXITED(status) ? WEXITSTATUS(status)
                                             : hipblaslt_bench::kRankChildFailed;
                pids[i]  = -1;
                ++exited;
            }

            if(exited == pids.size())
                break;
            if(std::chrono::steady_clock::now() > deadline)
                return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return true;
    }
} // namespace

TEST(FusedA2ARankEnv_smoke, RejectsLocalRankThatDoesNotMatchRank)
{
    hipblaslt_bench::LauncherEnv env;
    env.rank       = 0;
    env.world      = 2;
    env.local_rank = 1;

    Arguments arg;
    arg.init();
    arg.a2a_world = 2;

    EXPECT_FALSE(hipblaslt_bench::peers_reachable(env, arg));
}

using hipblaslt_bench::Collective;

// Worlds every fused collective is swept over, one GPU per rank. A host with
// fewer devices skips a world rather than failing it.
constexpr uint32_t kFusedWorlds[] = {2, 3, 4, 5, 6, 7, 8};

// Collectives the multi-process sweep runs.
constexpr Collective kFusedCollectives[] = {Collective::GemmA2A};

TEST(FusedA2ACollective_smoke, NamesRoundTrip)
{
    EXPECT_EQ(hipblaslt_bench::parse_collective(nullptr), Collective::GemmA2A);
    EXPECT_EQ(hipblaslt_bench::parse_collective(""), Collective::GemmA2A);
    EXPECT_EQ(hipblaslt_bench::parse_collective(
                  hipblaslt_bench::collective_name(Collective::GemmA2A)),
              Collective::GemmA2A);
    EXPECT_EQ(hipblaslt_bench::parse_collective("bias"), Collective::Unknown);
}

// One rule has to give a legal fused-A2A problem at every world, with the same
// shard per rank and the same local tail.
TEST(FusedA2AWorldShape_smoke, EveryWorldFrom2To8IsTileDivisible)
{
    for(uint32_t world : kFusedWorlds)
    {
        const auto    shape = hipblaslt_bench::fused_a2a_shape_for_world(world);
        const int64_t tile  = hipblaslt_bench::kFusedA2AMaxMacroTile0;
        EXPECT_EQ(shape.extent % (tile * int64_t(world)), 0) << "world " << world;
        EXPECT_EQ(shape.features % tile, 0) << "world " << world;
        EXPECT_EQ(shape.extent / int64_t(world), hipblaslt_bench::kFusedA2AShardFeatures)
            << "world " << world;
        EXPECT_EQ(shape.features - shape.extent, hipblaslt_bench::kFusedA2ALocalFeatures)
            << "world " << world;
    }

    const auto two = hipblaslt_bench::fused_a2a_shape_for_world(2);
    EXPECT_EQ(two.features, 4096);
    EXPECT_EQ(two.extent, 2048);
}

class FusedA2AMultiProcess_multi_gpu
    : public ::testing::TestWithParam<std::tuple<Collective, uint32_t>>
{
};

// Four launches over two channels walk 0, 1, 0, 1, so each channel is reused
// once while every rank is a peer of the other.
TEST_P(FusedA2AMultiProcess_multi_gpu, ReusedChannelsStayCorrect)
{
    const Collective collective = std::get<0>(GetParam());
    const uint32_t   world      = std::get<1>(GetParam());

    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < int(world))
        GTEST_SKIP() << "needs " << world << " devices, found " << deviceCount;

    const int port = freeLoopbackPort();
    ASSERT_GT(port, 0) << "could not reserve a loopback port";

    std::vector<pid_t> pids(world, -1);
    for(uint32_t rank = 0; rank < world; ++rank)
    {
        const int spawned = spawnRank(
            rank, world, port, hipblaslt_bench::collective_name(collective), pids[rank]);
        if(spawned != 0)
        {
            pids[rank] = -1;
            killAll(pids);
            FAIL() << "posix_spawn for rank " << rank << " -> errno " << spawned;
        }
    }

    std::vector<int> codes;
    if(!waitForRanks(pids, codes, kRankTimeoutSec))
    {
        killAll(pids);
        FAIL() << "ranks did not finish within " << kRankTimeoutSec << "s";
    }

    for(uint32_t rank = 0; rank < world; ++rank)
        if(codes[rank] == hipblaslt_bench::kRankChildSkipped)
            GTEST_SKIP() << "rank " << rank << " reported the run as unsupported";

    for(uint32_t rank = 0; rank < world; ++rank)
        EXPECT_EQ(codes[rank], hipblaslt_bench::kRankChildPassed) << "rank " << rank;
}

INSTANTIATE_TEST_SUITE_P(
    World,
    FusedA2AMultiProcess_multi_gpu,
    ::testing::Combine(::testing::ValuesIn(kFusedCollectives), ::testing::ValuesIn(kFusedWorlds)),
    [](const ::testing::TestParamInfo<std::tuple<Collective, uint32_t>>& info) {
        const char* tag = std::get<0>(info.param) == Collective::GemmA2A ? "GemmA2A" : "Unknown";
        return std::string(tag) + "_" + std::to_string(std::get<1>(info.param)) + "gpu";
    });
