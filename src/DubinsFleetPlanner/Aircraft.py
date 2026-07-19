from dataclasses import dataclass,asdict
import typing

@dataclass
class ACStats:
    id          :int
    airspeed    :float # Speed in air referential
    climb       :float # Vertical change in elevation (both directions)
    turn_radius :float # Minimal turn radius
    
    def asdict(self) -> dict[str,int|float]:
        return asdict(self)

@dataclass
class ACOptions:
    """
    Per-aircraft options matching the `--pathgen-config` keys.
    """
    ac_id               :int
    extend_start        :typing.Optional[list[float]] = None
    extend_end          :typing.Optional[list[float]] = None
    straights_only      :typing.Optional[bool] = None
    line                :typing.Optional[bool] = None
    line_ratios         :typing.Optional[list[float]] = None
    fixed_radius        :typing.Optional[bool] = None
    samples             :typing.Optional[int] = None
    ellipse             :typing.Optional[float] = None
    ellipse_border_only :typing.Optional[bool] = None

    @staticmethod
    def from_dict(d:dict[str,typing.Any]) -> "ACOptions":
        return ACOptions(
            ac_id=d["ac_id"],
            extend_start=d.get("extend-start"),
            extend_end=d.get("extend-end"),
            straights_only=d.get("straights-only"),
            line=d.get("line"),
            line_ratios=d.get("line-ratios"),
            fixed_radius=d.get("fixed-radius"),
            samples=d.get("samples"),
            ellipse=d.get("ellipse"),
            ellipse_border_only=d.get("ellipse-border-only")
        )

    def asdict(self) -> dict[str,typing.Any]:
        output = {"ac_id": self.ac_id}
        if self.extend_start is not None:
            output["extend-start"] = self.extend_start
        if self.extend_end is not None:
            output["extend-end"] = self.extend_end
        if self.straights_only is not None:
            output["straights-only"] = self.straights_only
        if self.line is not None:
            output["line"] = self.line
        if self.line_ratios is not None:
            output["line-ratios"] = self.line_ratios
        if self.fixed_radius is not None:
            output["fixed-radius"] = self.fixed_radius
        if self.samples is not None:
            output["samples"] = self.samples
        if self.ellipse is not None:
            output["ellipse"] = self.ellipse
        if self.ellipse_border_only is not None:
            output["ellipse-border-only"] = self.ellipse_border_only
        return output