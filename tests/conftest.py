import pytest
from lightning_teco import SDAAAccelerator


def pytest_addoption(parser):
    parser.addoption("--sdaas", action="store", type=int,
                     default=1, help="Number of sdaas 1-8")


@pytest.fixture()
def arg_sdaas(request):
    return request.config.getoption("--sdaas")


@pytest.fixture()
def device_count(pytestconfig):
    arg_sdaas = int(pytestconfig.getoption("sdaas"))
    if not arg_sdaas:
        assert SDAAAccelerator.auto_device_count() >= 1
        return 1
    assert arg_sdaas <= SDAAAccelerator.auto_device_count(
    ), "More sdaa devices asked than present"
    return arg_sdaas
