import logging

import dirs

def create_logger(name, log_file, formatter, level=logging.INFO, print_console=False):
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    if print_console:
        logger.addHandler(logging.StreamHandler())

    return logger

def create_training_loggers():
    # info logger
    info_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    info_logger = create_logger('info_logger', dirs.out_path('trained', 'training_info.log'), 
                                info_formatter, print_console=True)

    # data logger
    data_formatter = logging.Formatter()
    data_logger = create_logger('data_logger', dirs.out_path('trained', 'training_errors.log'),
                                data_formatter)
    
    return info_logger, data_logger

def get_training_loggers():
    info_logger = logging.getLogger('info_logger')
    data_logger = logging.getLogger('data_logger')
    return info_logger, data_logger
