root@72170d0458e9:/home/DeepSpeed_woo# pytest -q tests/unit/runtime/test_precision_config_loss_scale.py
=================================================================== test session starts ===================================================================
platform linux -- Python 3.11.10, pytest-8.3.5, pluggy-1.6.0 -- /usr/bin/python
cachedir: .pytest_cache
Using --randomly-seed=1526199052
rootdir: /home/DeepSpeed_woo/tests
configfile: pytest.ini
plugins: xdist-3.8.0, randomly-4.0.1, forked-1.6.0, anyio-4.6.0
collected 10 items

tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_accepts_valid_values[3] PASSED                                         [ 10%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_accepts_valid_values[0] PASSED                                         [ 20%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_rejects_invalid_values[inf] PASSED                                     [ 30%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_accepts_valid_values[1] PASSED                                         [ 40%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_rejects_invalid_values[nan] PASSED                                     [ 50%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_accepts_valid_values[2.0] PASSED                                       [ 60%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_rejects_invalid_values[True] PASSED                                    [ 70%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_invalid_type_has_clear_error[loss_scale0] PASSED                       [ 80%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_rejects_invalid_values[-1] PASSED                                      [ 90%]
tests/unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_invalid_type_has_clear_error[loss_scale1] PASSED                       [100%]

==================================================================== warnings summary =====================================================================
../../usr/local/lib/python3.11/dist-packages/torch/jit/_script.py:362: 14 warnings
  /usr/local/lib/python3.11/dist-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_accepts_valid_values[3]
  /home/DeepSpeed_woo/tests/conftest.py:47: UserWarning: Running test without verifying torch version, please provide an expected torch version with --torch_ver
    warnings.warn(

unit/runtime/test_precision_config_loss_scale.py::test_fp16_loss_scale_accepts_valid_values[3]
  /home/DeepSpeed_woo/tests/conftest.py:54: UserWarning: Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
==================================================================== slowest durations ====================================================================

(30 durations < 1s hidden.  Use -vv to show these durations.)
============================================================= 10 passed, 16 warnings in 4.18s =============================================================
