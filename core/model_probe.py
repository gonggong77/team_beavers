"""모델 파일에서 클래스 이름과 학습 해상도를 자동으로 읽어낸다.

모델팀이 클래스 인덱스를 어떻게 정했는지 미리 몰라도 되게 만드는 장치다.
이름으로 자동 추정하고, 실패하면 사이드바에서 사람이 직접 고르게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RIP_HINTS = ("rip", "current", "ripcurrent", "rip_current", "이안")
PERSON_HINTS = ("person", "people", "human", "swimmer", "사람")


@dataclass
class ModelInfo:
    names: dict[int, str] = field(default_factory=dict)
    imgsz: int = 640
    rip_ids: list[int] = field(default_factory=list)
    person_ids: list[int] = field(default_factory=list)
    unmapped: list[int] = field(default_factory=list)

    @property
    def needs_manual_mapping(self) -> bool:
        return not self.rip_ids or not self.person_ids


def _match(name: str, hints: tuple[str, ...]) -> bool:
    lowered = str(name).lower().replace(" ", "").replace("-", "_")
    return any(h in lowered for h in hints)


def probe(detector) -> ModelInfo:
    names = dict(getattr(detector, "names", {}) or {})
    info = ModelInfo(names=names)

    model = getattr(detector, "model", None)
    if model is not None:
        for getter in (
            lambda: model.ckpt["train_args"]["imgsz"],
            lambda: model.overrides["imgsz"],
            lambda: model.args["imgsz"],
        ):
            try:
                value = getter()
                info.imgsz = int(value[0] if isinstance(value, (list, tuple)) else value)
                break
            except Exception:
                continue

    info.rip_ids = [i for i, n in names.items() if _match(n, RIP_HINTS)]
    info.person_ids = [i for i, n in names.items() if _match(n, PERSON_HINTS)]
    mapped = set(info.rip_ids) | set(info.person_ids)
    info.unmapped = [i for i in names if i not in mapped]

    # 클래스가 정확히 2개인데 이름 추정에 실패했다면 순서대로 가정해 둔다.
    # 어디까지나 초기값이고, 사이드바에서 사람이 확인해야 한다.
    if len(names) == 2 and info.needs_manual_mapping:
        keys = sorted(names)
        info.rip_ids = info.rip_ids or [keys[0]]
        info.person_ids = info.person_ids or [k for k in keys if k not in info.rip_ids]

    return info
