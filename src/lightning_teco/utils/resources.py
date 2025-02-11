# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, MutableSequence
from lightning_teco.lightning import TorchElasticEnvironment
from lightning_utilities.core.imports import package_available


from pytorch_lightning.utilities.exceptions import MisconfigurationException


def check_environment():
    if package_available("torch_sdaa"):
        import torch_sdaa
    else:
        raise ImportError("Please Install torch_sdaa!")

    sdaa_avaliable = torch_sdaa.backend.is_available()
    if not sdaa_avaliable:
        raise Exception(
            "Sdaa Device is not available, please check your environment!")


def _parse_sdaa_ids(
        sdaas: Optional[Union[int, str, List[int]]]) -> Optional[List[int]]:
    """
    Parses the SDAA IDs given in the format as accepted by the
    :class:`~pytorch_lightning.trainer.Trainer`.

    Args:
        sdaas: An int -1 or string '-1' indicate that all available SDAAs should be used.
            A list of unique ints or a string containing a list of comma separated unique integers
            indicates specific SDAAs to use.
            An int of 0 means that no SDAAs should be used.
            Any int N > 0 indicates that SDAAs [0..N) should be used.

    Returns:
        A list of SDAAs to be used or ``None`` if no SDAAs were requested

    Raises:
        MisconfigurationException:
            If no SDAAs are available but the value of sdaas variable indicates request for SDAAs
    """
    # Check that sdaas param is None, Int, String or Sequence of Ints
    _check_data_type(sdaas)

    # Handle the case when no SDAAs are requested
    if sdaas is None or (isinstance(sdaas, int) and sdaas == 0) or str(sdaas).strip() in ("0", "[]"):
        return None

    # We know the user requested SDAAs therefore if some of the
    # requested SDAAs are not available an exception is thrown.
    sdaas = _normalize_parse_sdaa_string_input(sdaas)
    sdaas = _normalize_parse_sdaa_input_to_list(sdaas)
    if not sdaas:
        raise MisconfigurationException(
            "SDAAs requested but none are available.")

    if (
        TorchElasticEnvironment.detect()
        and len(sdaas) != 1
        and len(_get_all_available_sdaas()) == 1
    ):
        # Omit sanity check on torchelastic because by default it shows one visible SDAA per process
        return sdaas

    # Check that SDAAs are unique. Duplicate SDAAs are not supported by the backend.
    _check_unique(sdaas)

    return _sanitize_sdaa_ids(sdaas)


def _normalize_parse_sdaa_string_input(s: Union[int, str, List[int]]) -> Union[int, List[int]]:
    if not isinstance(s, str):
        return s
    if s == "-1":
        return -1
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if len(x) > 0]
    return int(s.strip())


def _sanitize_sdaa_ids(sdaas: List[int]) -> List[int]:
    """Checks that each of the SDAAs in the list is actually available. Raises a MisconfigurationException if any of
    the SDAAs is not available.

    Args:
        sdaas: List of ints corresponding to SDAA indices

    Returns:
        Unmodified sdaas variable

    Raises:
        MisconfigurationException:
            If machine has fewer available SDAAs than requested.
    """

    all_available_sdaas = _get_all_available_sdaas()
    for sdaa in sdaas:
        if sdaa not in all_available_sdaas:
            raise MisconfigurationException(
                f"You requested sdaa: {sdaas}\n But your machine only has: {all_available_sdaas}"
            )
    return sdaas


def _normalize_parse_sdaa_input_to_list(
        sdaas: Union[int, List[int], Tuple[int, ...]]) -> Optional[List[int]]:
    assert sdaas is not None
    if isinstance(sdaas, (MutableSequence, tuple)):
        return list(sdaas)

    # must be an int
    if not sdaas:  # sdaas==0
        return None
    if sdaas == -1:
        return _get_all_available_sdaas()
    return list(range(sdaas))


def _get_all_available_sdaas() -> List[int]:
    """
    Returns:
        A list of all available SDAAs
    """
    return list(range(num_sdaa_devices()))


def _check_data_type(device_ids: Any) -> None:
    msg = "Device IDs (SDAA) must be an int, a string, a sequence of ints or None, but you passed"

    if device_ids is None:
        return
    elif isinstance(device_ids, (MutableSequence, tuple)):
        for id_ in device_ids:
            if type(id_) is not int:
                raise MisconfigurationException(
                    f"{msg} a sequence of {type(id_).__name__}.")
    elif type(device_ids) not in (int, str):
        raise MisconfigurationException(f"{msg} {type(device_ids).__name__}.")


def _check_unique(device_ids: List[int]) -> None:
    if len(device_ids) != len(set(device_ids)):
        raise MisconfigurationException("Device ID's (SDAA) must be unique.")


@lru_cache(1)
def num_sdaa_devices() -> int:
    import torch_sdaa
    return torch_sdaa.backend.device_count()
