"""
Curriculum manager — auto-advances difficulty based on rolling success rate.

easy → medium → hard, with retreat if performance drops.
The environment calls update() after each episode and reads current_config().
"""
from .config import EnvConfig, CurriculumConfig, RealismConfig, TaskConfig


_LEVELS = {
    "easy":   EnvConfig.easy(),
    "medium": EnvConfig.medium(),
    "hard":   EnvConfig.hard(),
}


class CurriculumManager:
    def __init__(self, cfg: CurriculumConfig):
        self._cfg = cfg
        self._level_idx = 0
        self._levels = cfg.levels
        self._advance_count = 0   # consecutive episodes above threshold
        self._retreat_count = 0

    @property
    def current_level(self) -> str:
        return self._levels[self._level_idx]

    def current_config(self) -> EnvConfig:
        return _LEVELS[self.current_level]

    def update(self, success_rate: float) -> str:
        """Call after each episode. Returns level name (may have changed)."""
        if not self._cfg.enabled:
            return self.current_level

        if success_rate >= self._cfg.advance_threshold:
            self._advance_count += 1
            self._retreat_count = 0
            if self._advance_count >= 5 and self._level_idx < len(self._levels) - 1:
                self._level_idx += 1
                self._advance_count = 0
                print(f"[Curriculum] Advanced to {self.current_level} "
                      f"(success_rate={success_rate:.0%})")
        elif success_rate <= self._cfg.retreat_threshold:
            self._retreat_count += 1
            self._advance_count = 0
            if self._retreat_count >= 3 and self._level_idx > 0:
                self._level_idx -= 1
                self._retreat_count = 0
                print(f"[Curriculum] Retreated to {self.current_level} "
                      f"(success_rate={success_rate:.0%})")
        else:
            self._advance_count = 0
            self._retreat_count = 0

        return self.current_level
