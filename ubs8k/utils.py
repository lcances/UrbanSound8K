import logging


def set_logs(loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)


def conditional_cache(func):
    def decorator(*args, **kwargs):
        if "filename" in kwargs.keys() and "cached" in kwargs.keys():
            filename = kwargs["filename"]
            cached = kwargs["cached"]

            if filename is not None and cached:
                if filename not in decorator.cache.keys():
                    decorator.cache[filename] = func(*args, **kwargs)
                    return decorator.cache[filename]

                else:
                    if decorator.cache[filename] is None:
                        decorator.cache[filename] = func(*args, **kwargs)
                        return decorator.cache[filename]
                    else:
                        return decorator.cache[filename]

        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator
