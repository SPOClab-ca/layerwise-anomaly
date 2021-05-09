import time

def timed_func(foo):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        results = foo(*args, **kwargs)
        print ("{} done in {:.2f} seconds.".format(foo.__name__, time.time() - start_time))
        return results
    return wrapper