import inspect
from typing import Any, Dict

FromStore = Any


class ToStore(dict):
  pass


def update(func: Any, store: Dict[str, Any]) -> Any:
  stored_values = {}
  sig = inspect.signature(func)
  pars = sig.parameters
  new_pars = {}

  for item in pars.items():
    key = item[0]
    if item[1].annotation == FromStore:
      stored_values[key] = store[key]
    else:
      new_pars[key] = pars[key]

  def called(*args, **kwargs):
    res = func(*args, **kwargs, **stored_values)
    if type(res) == ToStore:
      res = {**store, **res}
    return res

  an = sig.return_annotation
  called.__signature__ = inspect.Signature(list(new_pars.values()),
                                           return_annotation=an)
  return called
