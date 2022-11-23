import logging
import logging.config


class Logger(object):
    def __init__(self, config, name):
        logging.config.dictConfig(config)
        self.logger = logging.getLogger(name)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
    
    def set_verbosity(self, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'\
                        .format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        self.logger.setLevel(self.log_levels[verbosity])
    
    def warning(self, warning_str):
        self.logger.warning(warning_str)
    
    def info(self, info_str):
        self.logger.info(info_str)
    
    def debug(self, debug_str):
        self.logger.debug(debug_str)
