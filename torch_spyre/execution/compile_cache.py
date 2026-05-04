# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from contextlib import contextmanager

from hashlib import sha256
from torch._inductor.runtime.cache_dir_utils import cache_dir
from torch_spyre._inductor.op_spec import OpSpec
from torch._inductor.utils import (
    clear_on_fresh_cache,
)


@clear_on_fresh_cache
class SpyreCompileCache:
    __cache_built: bool = False
    __cache_dir: str = ""
    __cache_hits: int = 0

    @staticmethod
    def get_cache_location():
        return os.path.join(cache_dir(), "inductor-spyre")

    @classmethod
    def build_cache(cls):
        if cls.__cache_built:
            return
        cls.__cache_dir: str = SpyreCompileCache.get_cache_location()
        if not Path(cls.__cache_dir).exists():
            Path(cls.__cache_dir).mkdir(parents=True, exist_ok=True)
        cls.__cache_built = True

    @classmethod
    def try_load(cls, specs: list[OpSpec]) -> tuple[str | None, str]:
        # Create a hash of the specs
        cls.build_cache()
        specs_hash = sha256(str(specs).encode()).hexdigest()
        cache_dir_loc = Path(cls.__cache_dir) / specs_hash
        # test if the expected artifact exists in the cache directory, if not treat as cache miss
        if SpyreCompileCache.artifacts_exist(str(cache_dir_loc)):
            cls.__cache_hits += 1
            return (str(cache_dir_loc), specs_hash)
        return (None, specs_hash)

    @classmethod
    def cache_hits(cls) -> int:
        return cls.__cache_hits

    @classmethod
    def cache_clear(cls):
        cls.__cache_built = False
        cls.__cache_hits = 0

    @classmethod
    def cache_dir(cls) -> str:
        return cls.__cache_dir

    @staticmethod
    def artifacts_exist(cache_dir_loc: str) -> bool:
        return (Path(cache_dir_loc) / "g2.graph.cbor").exists()


@contextmanager
def cache_context(op_specs: list[OpSpec], specs_hash: str | None = None):
    spyre_cache = get_spyre_cache()
    dir_hash = (
        specs_hash
        if specs_hash is not None
        else sha256(str(op_specs).encode()).hexdigest()
    )  # Create a unique directory name based on the specs
    spec_cache_dir = Path(spyre_cache.cache_dir()) / dir_hash
    spec_cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield str(spec_cache_dir)
    finally:
        pass


spyre_cache = SpyreCompileCache()


def get_spyre_cache() -> SpyreCompileCache:
    spyre_cache.build_cache()
    return spyre_cache
