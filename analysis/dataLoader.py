from dataclasses import dataclass, field
from typing import List, Dict, Union, Any, Optional, TypeAlias


@dataclass(frozen=True)
class GpuDevice:
    name: str


InnerOption = Union[List[str], Dict[str, Dict[str, Any]]]


@dataclass(frozen=True)
class OptionContainer:
    content: InnerOption = field(default_factory=dict)

    @property
    def is_list(self) -> bool:
        return isinstance(self.content, list)

    @property
    def is_dict(self) -> bool:
        return isinstance(self.content, dict)


@dataclass(frozen=True)
class Impl:
    impl_name: str
    preset_name: str
    preset_description: str
    options: OptionContainer


@dataclass(frozen=True)
class Recipe:
    backend: Impl
    benchmark: Impl
    case_: Impl
    stats: Impl
    stopping_criterion: Impl
    suite: Optional[Impl]


@dataclass(frozen=True)
class ResultFileMetadata:
    datetime: str
    git_version: str
    baseliner_version: str


@dataclass(frozen=True)
class BenchmarkRunSetup:
    datetime: str
    git_version: str
    baseliner_version: str
    run_id: str
    device: GpuDevice
    recipes: Recipe


ScalarMetric: TypeAlias = Union[float, int, str, None]


@dataclass
class ConfidenceInterval:
    low: ScalarMetric
    high: ScalarMetric


VectorMetric: TypeAlias = List[ScalarMetric]


@dataclass
class Metric:
    value: Union[ScalarMetric, VectorMetric, ConfidenceInterval]
    name: str
    unit: Optional[str] = None  # Some metrics (like repetitions) have no unit

    def is_vector(self) -> bool:
        return isinstance(self.value, list)

    def is_confidence_interval(self) -> bool:
        return isinstance(self.value, ConfidenceInterval)

    def is_scalar(self) -> bool:
        return isinstance(self.value, ScalarMetric)


@dataclass
class BenchmarkResult:
    metrics: Dict[str, Metric] = field(default_factory=dict)
