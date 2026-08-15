#pragma once

#include "spdlog/spdlog.h"

namespace lumia {
    class Logger {
    public:
        static void init();
        static std::shared_ptr<spdlog::logger> get();
        static void set_level(spdlog::level::level_enum level);
        static void shutdown();
    private:
        inline static std::shared_ptr<spdlog::logger> logger_ = nullptr;
    };
}

#define LUMIA_LOGGER_INIT() ::lumia::Logger::init()

#define LUMIA_TRACE(...) \
::lumia::Logger::get()->trace(__VA_ARGS__)

#define LUMIA_DEBUG(...) \
::lumia::Logger::get()->debug(__VA_ARGS__)

#define LUMIA_INFO(...) \
::lumia::Logger::get()->info(__VA_ARGS__)

#define LUMIA_WARN(...) \
::lumia::Logger::get()->warn(__VA_ARGS__)

#define LUMIA_ERROR(...) \
::lumia::Logger::get()->error(__VA_ARGS__)

#define LUMIA_CRITICAL(...) \
::lumia::Logger::get()->critical(__VA_ARGS__)
