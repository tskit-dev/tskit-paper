import util

VECTORISED = False


def run(ts_path, max_sites):
    return util.benchmark_external(
        f"compiled_implementations/cpp_sequential", ts_path, max_sites
    )
