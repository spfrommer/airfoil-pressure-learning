import logging

from airsim import dirs

def create_logger(name, log_file, formatter, mode, level=logging.INFO, print_console=False):
    handler = logging.FileHandler(log_file, mode=mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    if print_console:
        logger.addHandler(logging.StreamHandler())

    return logger

def create_loggers(training=True, append=False):
    mode = 'a' if append else 'w'
    write_dir = 'training' if training else 'testing'

    # info logger
    info_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    info_logger = create_logger('info_logger', dirs.out_path(write_dir, 'info.log'), 
                                info_formatter, mode, print_console=False)

    # data logger
    data_formatter = logging.Formatter()
    data_logger = create_logger('data_logger', dirs.out_path(write_dir, 'data.log'),
                                data_formatter, mode)
    
    return info_logger, data_logger

def get_loggers():
    info_logger = logging.getLogger('info_logger')
    data_logger = logging.getLogger('data_logger')
    return info_logger, data_logger
