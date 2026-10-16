from dataclasses import dataclass
import math


@dataclass
class SpeedProfile:
    """
    Class for speed profile with constant acceleration.
    """

    base_speed: float = 1.0
    acceleration: float = 0.0

    @classmethod
    def from_start_end_duration(
        cls,
        start_speed: float,
        end_speed: float,
        duration: float,
    ) -> "SpeedProfile":
        """Construct a profile from start speed, end speed, and duration."""
        return cls(
            base_speed=start_speed,
            acceleration=(end_speed - start_speed) / duration,
        )

    @classmethod
    def from_length(
        cls,
        start_speed: float,
        end_speed: float,
        length: float,
    ) -> "SpeedProfile":
        """Construct a profile from start speed, end speed, and distance."""
        acceleration = (
            end_speed * end_speed - start_speed * start_speed
        ) / (2 * length)

        return cls(
            base_speed=start_speed,
            acceleration=acceleration,
        )

    def get_speed_at_time(self, time: float) -> float:
        """Return the speed at the specified time."""
        return self.base_speed + self.acceleration * time

    def get_length_at_time(self, time: float) -> float:
        """Return the distance traveled at the specified time."""
        return (
            0.5 * self.acceleration * time * time
            + self.base_speed * time
        )

    def get_time_at_length(self, length: float) -> float:
        """Return the first non-negative time at which the distance is reached.

        Returns NaN if there is no non-negative real solution.
        """
        if self.acceleration == 0:
            if self.base_speed == 0:
                # Match C++ floating-point division behavior.
                if length == 0:
                    return math.nan
                return math.copysign(math.inf, length)

            return length / self.base_speed

        discriminant = (
            self.base_speed * self.base_speed
            + 2 * self.acceleration * length
        )

        if discriminant < 0:
            return math.nan

        sqrt_discriminant = math.sqrt(discriminant)

        t1 = (
            -self.base_speed + sqrt_discriminant
        ) / self.acceleration

        t2 = (
            -self.base_speed - sqrt_discriminant
        ) / self.acceleration

        if t1 >= 0:
            return t1
        elif t2 >= 0:
            return t2
        else:
            return math.nan

    def shift_by_time(self, time: float) -> None:
        """Shift the profile's time origin while preserving acceleration."""
        self.base_speed = self.get_speed_at_time(time)

    def shifted_by_time(self, time: float) -> "SpeedProfile":
        """Return a shifted copy without modifying the original profile."""
        output = SpeedProfile(
            base_speed=self.base_speed,
            acceleration=self.acceleration,
        )
        output.shift_by_time(time)
        return output