import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm as tvm
from tvm.target import Target


def _get_issue_735_source(pass_configs=None) -> str:
    rows, group = 6, 3

    @T.prim_func
    def main(A: T.Tensor((rows, 128), T.float32), B: T.Tensor((rows, 128), T.float32)):
        with T.Kernel(rows, threads=128) as bx:
            tx = T.get_thread_binding()
            # A row index that divides by a constant that is not a power of two,
            # as a grouped order of the blocks computes it.
            row = bx % group * (rows // group) + bx // group
            for i in T.serial(4):
                B[row, tx] = B[row, tx] + A[row, tx] * i

    target = Target("cuda")
    with tvm.transform.PassContext(config=pass_configs or {}), target:
        artifact = tilelang.lower(main, target=target)
    return artifact.kernel_source


@tilelang.testing.requires_cuda
def test_repeated_index_is_computed_once():
    src = _get_issue_735_source()
    assert src.count("% 3") == 1
    assert src.count("/ 3") == 1


@tilelang.testing.requires_cuda
def test_disable_cse_keeps_each_index_inlined():
    src = _get_issue_735_source({tilelang.PassConfigKey.TIR_DISABLE_CSE: True})
    assert src.count("% 3") > 1


if __name__ == "__main__":
    tilelang.testing.main()
