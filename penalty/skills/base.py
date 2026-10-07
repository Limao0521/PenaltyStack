"""Common contract for bounded skill parameters."""
from dataclasses import asdict, fields
import math


class ParameterizedSkill:
    """`bounds()` lists only the parameters the motion runtime honours.

    Other fields keep their nominal B-Human value; `apply` rejects any
    deviation instead of silently ignoring it.
    """
    parameters = None  # dataclass type
    BOUNDS = {}
    RUNTIME_SUPPORTED = ()

    def __init__(self):
        self.reset()

    def reset(self):
        self.current = self.parameters()

    def bounds(self):
        return {name: self.BOUNDS[name] for name in self.RUNTIME_SUPPORTED}

    @classmethod
    def validate(cls, params):
        if not isinstance(params, cls.parameters):
            raise TypeError('se esperaba %s' % cls.parameters.__name__)
        nominal = cls.parameters()
        for field in fields(params):
            value = getattr(params, field.name)
            low, high = cls.BOUNDS[field.name]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('%s debe ser finito' % field.name)
            if not low <= value <= high:
                raise ValueError('%s fuera de [%g, %g]' % (field.name, low, high))
            if (field.name not in cls.RUNTIME_SUPPORTED
                    and value != getattr(nominal, field.name)):
                raise ValueError('%s no está expuesto por el runtime B-Human actual' % field.name)
        return params

    def apply(self, params):
        self.current = self.validate(params)
        return self.request()

    def request(self):
        raise NotImplementedError

    def record(self):
        return {'parameters': asdict(self.current), 'runtime_supported': list(self.RUNTIME_SUPPORTED)}

    @classmethod
    def from_vector(cls, values, names=None):
        names = tuple(names or cls.RUNTIME_SUPPORTED)
        if len(values) != len(names):
            raise ValueError('vector y nombres de parámetros no coinciden')
        return cls.parameters(**{n: float(v) for n, v in zip(names, values)})
