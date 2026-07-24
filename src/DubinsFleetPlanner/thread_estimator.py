import psutil

def thread_count() -> int:
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count = psutil.cpu_count()
    
    if cpu_count_physical is None or cpu_count is None:
        return 8
    else:
        return cpu_count // cpu_count_physical