import logging

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
    # logging.basicConfig()

    # info logger
    info_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    info_logger = create_logger('info_logger', 'training_info.log', info_formatter, print_console=True)

    # data logger
    data_formatter = logging.Formatter()
    data_logger = create_logger('data_logger', 'training_errors.csv', data_formatter)
    
    return info_logger, data_logger
