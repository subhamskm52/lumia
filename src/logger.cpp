#include "logger.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#ifdef LUMIA_LOG_TO_FILE
#include "spdlog/sinks/daily_file_sink.h"
#endif


namespace lumia {

    void Logger::init() {
        if (logger_) return;
        auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        logger_ =  std::make_shared<spdlog::logger>("lumia",console_sink);

        #ifdef LUMIA_LOG_TO_FILE
            auto file_sink =std::make_shared<spdlog::sinks::daily_file_sink_mt>("logs/lumia.log", 0, 0);
            logger_->sinks().push_back(file_sink);
        #endif

        logger_->set_level(spdlog::level::debug);
        logger_->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    }
    std::shared_ptr<spdlog::logger> Logger::get() {
        if (!logger_) init();
        return logger_;
    }
    void Logger::set_level(spdlog::level::level_enum level) {
        get()->set_level(level);
    }
    void Logger::shutdown() {
        if (!logger_) {
            return;
        }
        logger_->flush();
        logger_.reset();
    }
}
