import util

VECTORISED = True


def run(ts_path, max_sites, chunk_size):
    return util.benchmark_external(
        f"compiled_implementations/c_vectorised", ts_path, max_sites, chunk_size
    )
