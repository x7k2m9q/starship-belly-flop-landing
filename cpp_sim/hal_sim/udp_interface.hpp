// =============================================================================
// udp_interface.hpp - UDP 通信接口 (Phase 3 软硬件协同验证)
//
// 架构:
//   Python物理引擎 (dynamics.py) ←UDP→ C++飞控 (FreeRTOS任务模拟)
//
// 协议:
//   Python → C++: SensorData 结构体 (52字节, 二进制)
//   C++ → Python: ControlOutput 结构体 (64字节, 二进制)
//
// 端口:
//   28015: Python→C++ (传感器数据)
//   28016: C++→Python (控制输出)
//
// 数据格式 (与 types.hpp 严格对齐, 无填充):
//   SensorData:  gyro[3]*4 + accel[3]*4 + gps_pos[3]*4 + gps_vel[3]*4
//                + gps_valid*1 + radar_alt*4 + radar_valid*1 + timestamp*4
//                = 52字节 (紧凑)
//   ControlOutput: throttle*4 + q_des[4]*4 + omega_des[3]*4 + tvc[2]*4
//                  + gf[3]*4 + rcs[3]*4 + status*1 + n_engines*1
//                  + total_thrust*4 + timestamp*4
//                  = 72字节 (紧凑)
// =============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <cstdio>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    using socket_t = SOCKET;
    #define INVALID_SOCK INVALID_SOCKET
    #define CLOSE_SOCKET closesocket
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    using socket_t = int;
    #define INVALID_SOCK (-1)
    #define CLOSE_SOCKET close
#endif

#include "../core/types.hpp"

namespace falcon9 {

// ===========================================================================
// UDP初始化/清理 (Windows需要WSAStartup)
// ===========================================================================
inline bool udp_init() {
#ifdef _WIN32
    WSADATA wsa;
    int err = WSAStartup(MAKEWORD(2, 2), &wsa);
    if (err != 0) {
        printf("[UDP] WSAStartup失败: %d\n", err);
        return false;
    }
#endif
    return true;
}

inline void udp_cleanup() {
#ifdef _WIN32
    WSACleanup();
#endif
}

// ===========================================================================
// UdpReceiver - UDP接收器 (Python→C++, 传感器数据)
//
// 绑定端口 28015, 接收 SensorData 结构体
// 非阻塞模式, 超时100ms
// ===========================================================================
class UdpReceiver {
public:
    socket_t sock{INVALID_SOCK};
    sockaddr_in addr{};

    bool init(int port = 28015) {
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCK) {
            printf("[UDP] socket创建失败\n");
            return false;
        }

        // 非阻塞 + 接收超时
#ifdef _WIN32
        DWORD timeout = 100;  // 100ms
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#else
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<uint16_t>(port));

        if (bind(sock, (sockaddr*)&addr, sizeof(addr)) == -1) {
            printf("[UDP] bind失败 port=%d\n", port);
            CLOSE_SOCKET(sock);
            return false;
        }

        printf("[UDP] 接收器绑定 port=%d\n", port);
        return true;
    }

    // 接收 SensorData (非阻塞, 超时返回false)
    bool receive(SensorData& data) {
        char buf[sizeof(SensorData)];
        sockaddr_in from;
#ifdef _WIN32
        int from_len = sizeof(from);
#else
        socklen_t from_len = sizeof(from);
#endif
        int n = recvfrom(sock, buf, sizeof(buf), 0,
                         (sockaddr*)&from, &from_len);
        if (n != sizeof(SensorData)) {
            return false;  // 超时或数据不完整
        }

        memcpy(&data, buf, sizeof(SensorData));
        return true;
    }

    void close() {
        if (sock != INVALID_SOCK) {
            CLOSE_SOCKET(sock);
            sock = INVALID_SOCK;
        }
    }
};

// ===========================================================================
// UdpSender - UDP发送器 (C++→Python, 控制输出)
//
// 发送到端口 28016, 发送 ControlOutput 结构体
// ===========================================================================
class UdpSender {
public:
    socket_t sock{INVALID_SOCK};
    sockaddr_in dest_addr{};

    bool init(const char* dest_ip = "127.0.0.1", int port = 28016) {
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCK) {
            printf("[UDP] socket创建失败\n");
            return false;
        }

        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(static_cast<uint16_t>(port));
#ifdef _WIN32
        if (inet_pton(AF_INET, dest_ip, &dest_addr.sin_addr) != 1) {
            printf("[UDP] 非法IP: %s\n", dest_ip);
            closesocket(sock);
            sock = INVALID_SOCKET;
            return false;
        }
#else
        if (!inet_aton(dest_ip, &dest_addr.sin_addr)) {
            printf("[UDP] 非法IP: %s\n", dest_ip);
            close(sock);
            sock = -1;
            return false;
        }
#endif

        printf("[UDP] 发送器目标 %s:%d\n", dest_ip, port);
        return true;
    }

    // 发送 ControlOutput
    bool send(const ControlOutput& output) {
        int n = sendto(sock, (const char*)&output, sizeof(output), 0,
                       (sockaddr*)&dest_addr, sizeof(dest_addr));
        return n == sizeof(output);
    }

    void close() {
        if (sock != INVALID_SOCK) {
            CLOSE_SOCKET(sock);
            sock = INVALID_SOCK;
        }
    }
};

// ===========================================================================
// 紧凑二进制协议 (用于Python端兼容)
//
// Python端使用 struct.pack/unpack 对应格式:
//   SensorData: '<3f3f3f3f?f?f I' (52字节)
//   ControlOutput: '<f4f3f2f3f3fBBfI' (72字节)
//
// 注意: C++结构体可能有padding, Python端需用struct手动打包
// ===========================================================================

}  // namespace falcon9
