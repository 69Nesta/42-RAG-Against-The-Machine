class MathUtils:
    @staticmethod
    def is_in_range(number: int, origin: int, pourcent: float) -> bool:
        delta = int(abs(origin) * pourcent / 100)

        return number in range(origin - delta, origin + delta + 1)
